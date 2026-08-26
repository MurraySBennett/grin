# Deployment Guide

How to get GRIN live at `https://grin.murraysbennett.com`.

Your personal site at `murraysbennett.com` and paddle-exp at
`paddle.murraysbennett.com` are both completely untouched by any of this —
separate bucket, separate distribution, separate DNS record.

Everything below is done through the AWS Console in a browser — no AWS CLI,
no scripts to execute, nothing your laptop's admin restrictions can block.

---

## Overview — and why this looks nothing like paddle-exp's setup

```
grin.murraysbennett.com → CloudFront → S3   (that's the whole stack)
```

Paddle-exp needed a real server: it holds WebSocket connections open, runs a
Python process, and writes participant data to disk. GRIN has none of that —
every page is static HTML/CSS/JS, and the neural network runs _in the
visitor's own browser_. There's no EC2 instance here, nothing to patch or
reboot, and the whole "ongoing operations" story after this one-time setup
is: push to `main`, GitHub does the rest.

---

## Part 1: One-time AWS setup, in the Console

Do these five sections in order — later ones reference values created in
earlier ones. Where a value gets created (a bucket name, an ARN, a
distribution ID), write it down; you'll need several of them again in Part 2.

### 1a. The S3 bucket

1. **AWS Console → S3 → Create bucket**
2. Bucket name: `grin-murraysbennett`
3. AWS Region: **US East (N. Virginia) `us-east-1`**
4. Object Ownership: leave as **ACLs disabled** (the default)
5. **Block Public Access settings**: leave all four boxes **checked** (the
   default). This is deliberate, not an oversight — the bucket should never
   be directly public; CloudFront will read from it privately in 1c below.
6. Leave everything else at its default (Bucket Versioning off, default
   encryption on)
7. **Create bucket**

### 1b. The ACM certificate

**Before anything else in this section: check the region dropdown in the
top-right corner of the console says "N. Virginia" (`us-east-1`).** This is
the single easiest thing to get wrong here — certificates used by
CloudFront _must_ be requested in `us-east-1`, regardless of which region
your bucket or anything else lives in. If you request it in the wrong
region, CloudFront simply won't be able to find it later, with no obvious
error pointing back to this step.

1. **AWS Console → Certificate Manager → Request a certificate**
2. Choose **Request a public certificate → Next**
3. Fully qualified domain name: `grin.murraysbennett.com`
4. Validation method: **DNS validation** (leave selected)
5. Key algorithm: leave default (RSA 2048)
6. **Request**
7. Click into the certificate you just requested. Under the **Domains**
   section, since your DNS already lives in Route53, you should see a
   **"Create records in Route 53"** button — click it, then **Create
   records** on the confirmation screen. This does the DNS validation record
   creation for you automatically; you do not need to touch Route53
   manually for this part.
8. Wait for the certificate's status to change from **Pending validation**
   to **Issued** — usually a few minutes, sometimes up to 30. Refresh the
   page to check; there's nothing to click while waiting.

### 1c. The CloudFront distribution (this also creates the Origin Access Control)

1. **AWS Console → CloudFront → Create distribution**
2. **Origin domain**: click into the field and select your bucket
   (`grin-murraysbennett...s3.us-east-1.amazonaws.com`) from the dropdown —
   pick the plain S3 REST endpoint, not anything with "website" in the name
3. **Origin access**: select **"Origin access control settings
   (recommended)"**
4. Click **"Create control setting"** in the panel that appears → give it
   any name (e.g. `grin-oac`) → leave Signing behavior as **"Sign requests
   (recommended)"** → **Create**
5. Back on the main form, confirm the OAC you just created is selected
6. **Viewer protocol policy**: Redirect HTTP to HTTPS
7. **Allowed HTTP methods**: GET, HEAD
8. **Cache policy**: CachingOptimized
9. Scroll to **Settings**:
   - **Alternate domain name (CNAME)**: click **Add item**, enter
     `grin.murraysbennett.com`
   - **Custom SSL certificate**: select `grin.murraysbennett.com` from the
     dropdown. If it isn't there, the certificate from 1b either isn't
     validated yet or wasn't requested in `us-east-1` — go back and check
     before continuing.
   - **Default root object**: `index.html`
   - **Price class**: "Use only North America and Europe" (cheapest option
     that still covers where this site's visitors are)
10. **Create distribution**
11. **Copy the Distribution ID and Domain Name shown on the result page** —
    you'll need both later.
12. You should see a banner along the lines of **"The S3 bucket policy needs
    to be updated"** with a **Copy policy** button. Click it. This is
    CloudFront generating the exact bucket policy that lets _this specific
    distribution_ — not the internet, not any other CloudFront distribution
    — read from your bucket.
13. Go to **S3 → grin-murraysbennett → Permissions tab → Bucket policy →
    Edit**, paste what you copied, **Save changes**.

    _If that banner didn't appear for any reason_, paste this instead,
    replacing `YOUR_DISTRIBUTION_ARN` with the ARN from the distribution's
    **General** tab (looks like
    `arn:aws:cloudfront::123456789012:distribution/E1ABCDEFGHIJ`):

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "AllowCloudFrontServicePrincipalReadOnly",
          "Effect": "Allow",
          "Principal": { "Service": "cloudfront.amazonaws.com" },
          "Action": "s3:GetObject",
          "Resource": "arn:aws:s3:::grin-murraysbennett/*",
          "Condition": {
            "StringEquals": { "AWS:SourceArn": "YOUR_DISTRIBUTION_ARN" }
          }
        }
      ]
    }
    ```

14. Wait for the distribution's status to change from **Deploying** to
    **Enabled** — 5-15 minutes, nothing to click while waiting.

### 1d. The Route53 record

1. **AWS Console → Route 53 → Hosted zones → murraysbennett.com → Create
   record**
2. Record name: `grin`
3. Record type: **A**
4. Toggle **Alias** to **on**
5. Route traffic to: **Alias to CloudFront distribution**
6. Select your distribution from the list (it may take a minute after
   creation to appear here — if it's not showing yet, wait for 1c's
   "Enabled" status first)
7. Routing policy: **Simple routing**
8. **Create records**

### 1e. A scoped IAM user for GitHub Actions

This is the identity the GitHub Action authenticates as — deliberately
narrow: it can only touch this one bucket and invalidate this one
distribution, nothing else in your AWS account.

1. **AWS Console → IAM → Users → Create user**
2. User name: `grin-deploy`
3. Leave **"Provide user access to the AWS Management Console"** unchecked
   — this user only ever needs programmatic access, never a login
4. Click through **Next** without attaching any managed policies yet →
   **Create user**
5. Click into the `grin-deploy` user you just created → **Permissions** tab
   → **Add permissions → Create inline policy**
6. Switch to the **JSON** tab (instead of the visual editor) and paste,
   filling in your actual bucket name and the distribution ARN from 1c:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "S3ReadWriteThisBucketOnly",
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::grin-murraysbennett",
           "arn:aws:s3:::grin-murraysbennett/*"
         ]
       },
       {
         "Sid": "InvalidateThisDistributionOnly",
         "Effect": "Allow",
         "Action": "cloudfront:CreateInvalidation",
         "Resource": "YOUR_DISTRIBUTION_ARN"
       }
     ]
   }
   ```

7. **Next** → name the policy `grin-deploy-policy` → **Create policy**
8. Back on the user's page: **Security credentials** tab → **Access keys**
   section → **Create access key**
9. Choose **"Third-party service"** (or "Command line interface (CLI)" —
   either is fine, AWS just wants an acknowledgment that you understand this
   is a long-lived credential) → check the confirmation box → **Next** →
   (description is optional) → **Create access key**
10. **Copy the Access Key ID and Secret Access Key now, or download the
    provided `.csv`.** The secret key is shown exactly once — if you
    navigate away without copying it, you cannot retrieve it again, only
    generate a new one.

---

## Part 2: GitHub repository secrets

In the `grin` repo: **Settings → Secrets and variables → Actions → New
repository secret**. Add all five — some come straight from what you just
copied, some you're choosing yourself:

| Secret                       | Value                 |
| ---------------------------- | --------------------- |
| `AWS_ACCESS_KEY_ID`          | from Part 1e, step 10 |
| `AWS_SECRET_ACCESS_KEY`      | from Part 1e, step 10 |
| `AWS_REGION`                 | `us-east-1`           |
| `S3_BUCKET_NAME`             | `grin-murraysbennett` |
| `CLOUDFRONT_DISTRIBUTION_ID` | from Part 1c, step 11 |

These are separate from any secrets on your main site's repo or paddle-exp's
repo — GitHub secrets don't share across repositories even in the same
account.

---

## Part 3: before the first push

The deploy workflow (`.github/workflows/deploy.yaml`) has no build step — it
deploys exactly what's committed to `web/`, nothing generated on the fly.
Before pushing to `main`, confirm these are actually committed (not just
present on your machine). The workflow's smoke-check step also verifies them:

- [ ] `web/assets/models/cm/npe_model.onnx` (+ matching `manifest.json`)
- [ ] `web/assets/models/cm/recalibration.json` (optional; absent means the
      "calibrated intervals" toggle stays hidden rather than erroring)

The `cmrt` response-time model was withdrawn from the site: its weights come from a
generator retired on 2026-08-14 (`docs/dynamic_grt_rt_design.md`) and the replacement
is still in validation. The smoke-check no longer requires it. Re-add it to both the
checklist and the workflow's manifest loop when the replacement ships.

When you replace the release checkpoint, update the `.onnx` **and** the
manifest `version` / `artifact_sha256` / `training` fields together. Prefer
versioned filenames (e.g. `npe_model.v1.onnx`) if you want the long CloudFront
cache to stay safe across weight swaps.

---

## Part 4: ship it

```bash
git add -A
git commit -m "deploy: go live"
git push origin main
```

Watch it run under the **Actions** tab of the repo. Once it's green:

1. Visit `https://grin.murraysbennett.com`
2. Open DevTools → Network tab, reload
3. Specifically check for: no 404s on `ort-wasm-simd.wasm` or either `.onnx`
   file, and that Space Builder / Analyse actually complete an inference
   rather than sitting on "loading" forever

That combination catches the two failure modes most likely on a first real
deploy: a wrong content-type header and a model file that didn't make it
into the commit.

---

## Day-to-day operations

**Deploying a change:**

```bash
git add -A && git commit -m "description" && git push origin main
```

That's the whole thing. No server, no CLI, nothing else to run.

**Watching a deploy happen:**
GitHub repo → **Actions** tab → click the running workflow.

**Checking what's actually live right now:**
AWS Console → S3 → `grin-murraysbennett` → browse the objects directly.

**Forcing visitors to see a change immediately** (normally not needed — the
workflow already invalidates CloudFront on every deploy):
AWS Console → CloudFront → your distribution → **Invalidations** tab →
**Create invalidation** → path `/*` → **Create invalidation**.

**Rotating the deploy credentials** (if the access key ever leaks):
IAM → Users → `grin-deploy` → Security credentials → **Create access key**
(make the new one first) → update the two GitHub secrets → confirm a deploy
still works → **then** come back and delete the old key.

---

## Costs

| Resource                                            | Cost                                                                                                        |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| S3 storage                                          | A few cents/month — the whole site plus model weights is a few MB                                           |
| S3 requests                                         | Negligible at this traffic                                                                                  |
| CloudFront ("North America and Europe" price class) | Free tier covers ~1TB/month egress; effectively $0 for a personal-scale site                                |
| ACM certificate                                     | Free, always, for certificates used with CloudFront                                                         |
| Route53                                             | $0.50/month per hosted zone — you already have this for `murraysbennett.com`, this doesn't add a second one |

Total: a few cents a month, no matter how much traffic — there's no idle
server burning money the way an EC2 instance does even at zero visitors.

---

## Troubleshooting

**Site shows CloudFront's default error page, not GRIN:**
The distribution is probably still deploying — check its status on the
CloudFront console (**Deploying** vs **Enabled**). First deployment after
creation can take up to 15-20 minutes; this is normal, not a failure.

**`403 Forbidden` on every page:**
Almost always the bucket policy from Part 1c not matching the actual
distribution — re-check **S3 → grin-murraysbennett → Permissions → Bucket
policy**, and confirm the `AWS:SourceArn` value matches the distribution's
actual ARN exactly.

**`ort-wasm-simd.wasm` fails to load / model never finishes "loading":**
Check its `Content-Type` in DevTools → Network — should be
`application/wasm`. The deploy workflow sets this explicitly for exactly
this file, since the AWS CLI's default MIME-type guessing for `.wasm` is
unreliable depending on the runner's Python build.

**A deploy succeeded but the site looks unchanged:**
Browser cache, not a deploy failure — HTML is served `no-cache` so this
should self-correct on reload, but JS/CSS/assets are cached for an hour by
design (see the comment block in `deploy.yaml` for why). Hard-refresh
(Ctrl/Cmd+Shift+R) to confirm before assuming something's actually broken.

**GitHub Action fails with an AWS auth error:**
The `grin-deploy` access key may have been rotated or deleted. Check
**IAM → Users → grin-deploy → Security credentials** to see which keys are
currently active, and confirm the GitHub secrets match one of them.

**The certificate never shows up in CloudFront's dropdown (Part 1c, step 9):**
Almost always the `us-east-1` region mistake from Part 1b — check
**Certificate Manager**, confirm the region selector top-right, and confirm
the certificate's status is **Issued**, not **Pending validation**.
