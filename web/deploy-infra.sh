#!/usr/bin/env bash
# ============================================================================
# deploy-infra.sh — OPTIONAL. The primary setup path is now the AWS Console
# walkthrough in DEPLOYMENT.md Part 1 (no AWS CLI or script execution
# required). This script does the exact same thing via CLI instead, kept
# around in case CLI access becomes available later -- not needed for the
# console path, and not currently the recommended route.
#
# One-time AWS setup for grin.murraysbennett.com
#
# Creates: S3 bucket, ACM certificate, CloudFront distribution (with Origin
# Access Control, not a public bucket), the Route53 records, and a scoped
# IAM user for GitHub Actions to deploy as.
#
# This is DIFFERENT infrastructure from paddle-exp on purpose: GRIN is a
# static site with nothing server-side, ever, so there's no EC2 instance, no
# Docker, no Nginx, no Certbot here — S3+CloudFront serves the files directly
# and ACM handles the certificate for free, with no server to patch or reboot.
#
# Run this from a machine with the AWS CLI installed and configured with
# credentials that can create S3 buckets, ACM certs, CloudFront distributions,
# Route53 records, and IAM users. Takes ~15-20 minutes total, most of it
# waiting for ACM validation and CloudFront deployment (both unattended).
#
# Safe to re-run individual sections if something fails partway — each
# section prints the value it produced, so you can skip ahead manually by
# setting that variable yourself if a step already succeeded.
# ============================================================================
set -euo pipefail

# --- Configuration — the only things you should need to change ------------
BUCKET_NAME="grin-murraysbennett"
DOMAIN="grin.murraysbennett.com"
ROOT_DOMAIN="murraysbennett.com"
BUCKET_REGION="us-east-2"   # the bucket's region is independent of anything
                             # else in this script -- CloudFront reads from S3
                             # buckets in any region. Only the ACM cert below
                             # is hard-locked to us-east-1 (a real CloudFront
                             # requirement); that's handled separately and
                             # explicitly throughout, not via this variable.
IAM_USER_NAME="grin-deploy"
# ---------------------------------------------------------------------------

echo "=== Part 1: S3 bucket ==="
if [ "$BUCKET_REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$BUCKET_REGION"
else
  aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$BUCKET_REGION" \
    --create-bucket-configuration LocationConstraint="$BUCKET_REGION"
fi

# Block all public access -- CloudFront will read via Origin Access Control
# (below), not a public bucket policy. Nobody should ever be able to hit the
# bucket directly; only CloudFront should be able to.
aws s3api put-public-access-block --bucket "$BUCKET_NAME" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "Bucket created: $BUCKET_NAME"
echo

echo "=== Part 2: ACM certificate (must be requested in us-east-1 for CloudFront, regardless of bucket region) ==="
CERT_ARN=$(aws acm request-certificate \
  --domain-name "$DOMAIN" \
  --validation-method DNS \
  --region us-east-1 \
  --query CertificateArn --output text)
echo "Certificate requested: $CERT_ARN"

# ACM needs a moment to generate the validation record after the request
sleep 10

VALIDATION_NAME=$(aws acm describe-certificate --certificate-arn "$CERT_ARN" --region us-east-1 \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord.Name' --output text)
VALIDATION_VALUE=$(aws acm describe-certificate --certificate-arn "$CERT_ARN" --region us-east-1 \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord.Value' --output text)
echo "Validation record needed: $VALIDATION_NAME -> $VALIDATION_VALUE"
echo

echo "=== Part 3: create the ACM validation record in Route53 ==="
ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name "$ROOT_DOMAIN" \
  --query 'HostedZones[0].Id' --output text | sed 's|/hostedzone/||')
echo "Route53 zone for $ROOT_DOMAIN: $ZONE_ID"

cat > /tmp/grin-cert-validation.json <<EOF
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "$VALIDATION_NAME",
      "Type": "CNAME",
      "TTL": 300,
      "ResourceRecords": [{ "Value": "$VALIDATION_VALUE" }]
    }
  }]
}
EOF
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch file:///tmp/grin-cert-validation.json
echo "Validation record created."
echo

echo "=== Part 4: waiting for ACM to validate (usually 1-5 minutes with Route53) ==="
aws acm wait certificate-validated --certificate-arn "$CERT_ARN" --region us-east-1
echo "Certificate validated."
echo

echo "=== Part 5: CloudFront Origin Access Control ==="
OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config \
  "Name=grin-oac,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
  --query 'OriginAccessControl.Id' --output text)
echo "OAC created: $OAC_ID"
echo

echo "=== Part 6: CloudFront distribution ==="
S3_DOMAIN="${BUCKET_NAME}.s3.${BUCKET_REGION}.amazonaws.com"

cat > /tmp/grin-distribution.json <<EOF
{
  "CallerReference": "grin-$(date +%s)",
  "Comment": "grin.murraysbennett.com -- GRIN static site",
  "Enabled": true,
  "DefaultRootObject": "index.html",
  "Aliases": { "Quantity": 1, "Items": ["$DOMAIN"] },
  "Origins": {
    "Quantity": 1,
    "Items": [{
      "Id": "grin-s3-origin",
      "DomainName": "$S3_DOMAIN",
      "OriginAccessControlId": "$OAC_ID",
      "S3OriginConfig": { "OriginAccessIdentity": "" }
    }]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "grin-s3-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"], "CachedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"] } },
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "Compress": true
  },
  "ViewerCertificate": {
    "ACMCertificateArn": "$CERT_ARN",
    "SSLSupportMethod": "sni-only",
    "MinimumProtocolVersion": "TLSv1.2_2021"
  },
  "PriceClass": "PriceClass_100"
}
EOF

DIST_JSON=$(aws cloudfront create-distribution --distribution-config file:///tmp/grin-distribution.json)
DIST_ID=$(echo "$DIST_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['Distribution']['Id'])")
DIST_ARN=$(echo "$DIST_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['Distribution']['ARN'])")
DIST_DOMAIN=$(echo "$DIST_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['Distribution']['DomainName'])")
echo "Distribution created: $DIST_ID ($DIST_DOMAIN)"
echo

echo "=== Part 7: bucket policy -- allow ONLY this CloudFront distribution to read ==="
cat > /tmp/grin-bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowCloudFrontServicePrincipalReadOnly",
    "Effect": "Allow",
    "Principal": { "Service": "cloudfront.amazonaws.com" },
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::${BUCKET_NAME}/*",
    "Condition": { "StringEquals": { "AWS:SourceArn": "$DIST_ARN" } }
  }]
}
EOF
aws s3api put-bucket-policy --bucket "$BUCKET_NAME" --policy file:///tmp/grin-bucket-policy.json
echo "Bucket policy applied -- only this distribution can read from the bucket."
echo

echo "=== Part 8: Route53 alias record ==="
# Z2FDTNDATAQYW2 is not a placeholder to fill in -- it's the SAME fixed
# hosted-zone ID for every CloudFront distribution on every AWS account.
cat > /tmp/grin-dns-record.json <<EOF
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "$DOMAIN",
      "Type": "A",
      "AliasTarget": {
        "HostedZoneId": "Z2FDTNDATAQYW2",
        "DNSName": "$DIST_DOMAIN",
        "EvaluateTargetHealth": false
      }
    }
  }]
}
EOF
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch file:///tmp/grin-dns-record.json
echo "DNS record created: $DOMAIN -> $DIST_DOMAIN"
echo

echo "=== Part 9: scoped IAM user for GitHub Actions ==="
cat > /tmp/grin-deploy-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadWriteThisBucketOnly",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET_NAME}", "arn:aws:s3:::${BUCKET_NAME}/*"]
    },
    {
      "Sid": "InvalidateThisDistributionOnly",
      "Effect": "Allow",
      "Action": "cloudfront:CreateInvalidation",
      "Resource": "$DIST_ARN"
    }
  ]
}
EOF

aws iam create-user --user-name "$IAM_USER_NAME"
aws iam put-user-policy --user-name "$IAM_USER_NAME" \
  --policy-name grin-deploy-policy --policy-document file:///tmp/grin-deploy-policy.json
ACCESS_KEY_JSON=$(aws iam create-access-key --user-name "$IAM_USER_NAME")
ACCESS_KEY_ID=$(echo "$ACCESS_KEY_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['AccessKey']['AccessKeyId'])")
SECRET_ACCESS_KEY=$(echo "$ACCESS_KEY_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['AccessKey']['SecretAccessKey'])")

echo
echo "============================================================================"
echo " DONE. CloudFront is now deploying (takes 5-15 min before the site is live"
echo " at https://$DOMAIN -- DNS is already pointed at it, so it'll just start"
echo " working once CloudFront finishes, no further action needed for that part)."
echo
echo " Add these FIVE secrets to the grin GitHub repo now:"
echo " (Settings -> Secrets and variables -> Actions -> New repository secret)"
echo
echo "   AWS_ACCESS_KEY_ID          = $ACCESS_KEY_ID"
echo "   AWS_SECRET_ACCESS_KEY      = $SECRET_ACCESS_KEY"
echo "   AWS_REGION                 = $BUCKET_REGION"
echo "   S3_BUCKET_NAME              = $BUCKET_NAME"
echo "   CLOUDFRONT_DISTRIBUTION_ID = $DIST_ID"
echo
echo " The secret access key above is shown ONLY this once -- copy it now."
echo "============================================================================"