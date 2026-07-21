# Signing & Notarizing SpyDE for macOS

How to turn SpyDE's unsigned macOS build into a **signed + notarized** one so
users stop hitting the Gatekeeper "unidentified developer" wall — and so the
auto-update `.zip` is trusted too (electron-updater applies a notarized zip
without a warning).

> **Status: WIRED (2026-07-21).** `electron-builder.yml` has
> `hardenedRuntime`/`entitlements`/`notarize: true`, and `release.yml`'s mac leg
> imports the cert + stages the notarization key, gated on the job-level
> `MAC_SIGNING_ENABLED` flag (true when the `MAC_CERT_P12_BASE64` secret is set).
> The six secrets below are configured, so **the next release build signs +
> notarizes automatically**. A local `npm run dist` or a fork build with no
> secrets still produces an *unsigned* build (electron-builder logs "skipped macOS
> code signing") rather than failing. Steps 1–4 below are the one-time cert/secret
> setup (done); steps 5–7 document what's now wired + how to verify.

## What "signing an Apple app" means (two required steps)

1. **Code signing** — sign the `.app` / `.dmg` / `.zip` with your **Developer ID
   Application** certificate so macOS knows who built it.
2. **Notarization** — upload the signed build to Apple; they scan it and issue a
   *notarization ticket*, which is then **stapled** to the artifact so it
   validates offline.

Since macOS Catalina, **signing alone is not enough** — Gatekeeper requires
notarization too. You need both. electron-builder does the code signing during
`npm run dist`, and (with `notarize: true`) submits + staples automatically.

---

## Step 1 — Create the Developer ID Application certificate (once, on a Mac)

Distributing OUTSIDE the App Store → you need a **"Developer ID Application"**
cert (NOT "Mac App Distribution", NOT "Developer ID Installer" — that one is only
for `.pkg` installers, which SpyDE doesn't ship).

- Xcode → Settings → Accounts → your Apple ID → **Manage Certificates** → **+** →
  **Developer ID Application**.
- Or developer.apple.com → Certificates, IDs & Profiles → Certificates → **+** →
  "Developer ID Application".

This lands the cert + its private key in your login keychain.

### Which intermediate certificate?

Your signing cert chains up through Apple's **Developer ID** intermediate. Use the
current one — **"Developer ID Certification Authority (G2)"** — NOT the older
non-G2 "Developer ID Certification Authority" (still valid for old artifacts but
being retired; new certs chain to G2):

```
Apple Root CA
  └─ Developer ID Certification Authority (G2)     ← the intermediate
       └─ Developer ID Application: You (TEAMID)    ← your signing cert
```

**You normally don't pick this by hand** — Xcode / double-clicking the downloaded
cert installs the right intermediate automatically. It only matters if you hit a
"unable to build chain" / "CSSMERR_TP_NOT_TRUSTED" error, which on a **CI runner**
means the runner's keychain is missing the G2 intermediate. Guard against it by
exporting the full chain into the `.p12` (Step 2), or fetch it in CI:
`curl -sO https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer` then
`security import DeveloperIDG2CA.cer -k "$KEYCHAIN"`.

## Step 2 — Export the certificate as a `.p12` (for CI)

CI has no keychain, so export cert+key to a password-protected `.p12`:

1. Keychain Access → find **"Developer ID Application: Your Name (TEAMID)"**. To
   bake the intermediate into the `.p12` (so CI never has a broken chain),
   Cmd-click BOTH that cert **and** "Developer ID Certification Authority (G2)".
2. Right-click → **Export … (2 items)** → save `certificate.p12`, set an export
   password.
3. base64-encode it for a GitHub secret:
   ```bash
   base64 -i certificate.p12 -o certificate.p12.base64
   ```

> If your Keychain doesn't show the G2 intermediate, download it first:
> `curl -O https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer` and
> double-click to add it, then export.

## Step 3 — Create an App Store Connect API key (for notarization)

Preferred over an app-specific password (revocable, no 2FA prompts).

1. appstoreconnect.apple.com → Users and Access → **Integrations** tab →
   App Store Connect API → **+** → role **Developer**.
2. **Download the `.p8` — you can only download it once.** Note the **Key ID**
   and, at the top of that page, the **Issuer ID** (a UUID).

Your **Team ID** (10 chars) is on developer.apple.com → Membership.

## Step 4 — Add GitHub Actions secrets

Repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `MAC_CERT_P12_BASE64` | contents of `certificate.p12.base64` |
| `MAC_CERT_PASSWORD` | the `.p12` export password (Step 2) |
| `APPLE_API_KEY_P8` | full contents of the `.p8` file (Step 3) |
| `APPLE_API_KEY_ID` | the API Key ID |
| `APPLE_API_ISSUER_ID` | the Issuer ID (UUID) |
| `APPLE_TEAM_ID` | your 10-char Team ID |

## Step 5 — `electron-builder.yml` changes

Replace the `mac:` block's `identity: null` line with the signing + notarize
config (electron-builder 26 uses `notarize: true`, reading the API-key env vars
below):

```yaml
mac:
  category: public.app-category.education
  target: [dmg, zip]
  icon: ../spyde/Spyde.png
  hardenedRuntime: true                      # REQUIRED for notarization
  gatekeeperAssess: false                    # don't run Gatekeeper on the build host
  entitlements: build/entitlements.mac.plist
  entitlementsInherit: build/entitlements.mac.plist
  notarize: true                             # electron-builder submits + staples
  # (remove `identity: null` — that line DISABLES signing)
```

The entitlements file already exists at `build/entitlements.mac.plist`. Its
`disable-library-validation` key is **load-bearing for SpyDE**: the app spawns a
`uv`-managed Python subprocess with third-party native wheels (torch, numba,
rosettasciio readers) that are NOT signed by your Developer ID — without that
entitlement the hardened runtime refuses to load them and the backend won't start.

## Step 6 — `release.yml` changes (macOS build leg)

The build job is a 3-OS matrix; gate the signing steps to macOS with
`if: runner.os == 'macOS'`. Add BEFORE the "Build + publish desktop app" step:

```yaml
      - name: Import Apple signing certificate (macOS only)
        if: runner.os == 'macOS'
        env:
          MAC_CERT_P12_BASE64: ${{ secrets.MAC_CERT_P12_BASE64 }}
          MAC_CERT_PASSWORD: ${{ secrets.MAC_CERT_PASSWORD }}
        run: |
          set -euo pipefail
          KEYCHAIN=build.keychain
          KEYCHAIN_PW=$(openssl rand -base64 24)
          echo "$MAC_CERT_P12_BASE64" | base64 --decode > cert.p12
          security create-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
          security set-keychain-settings -lut 21600 "$KEYCHAIN"
          security unlock-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
          security import cert.p12 -k "$KEYCHAIN" -P "$MAC_CERT_PASSWORD" \
            -T /usr/bin/codesign -T /usr/bin/security
          # Belt-and-braces: ensure the Developer ID G2 intermediate is present so
          # codesign can build the chain even if the .p12 didn't include it.
          curl -fsSO https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer
          security import DeveloperIDG2CA.cer -k "$KEYCHAIN" || true
          security set-key-partition-list -S apple-tool:,apple:,codesign: \
            -s -k "$KEYCHAIN_PW" "$KEYCHAIN"
          security list-keychains -d user -s "$KEYCHAIN" login.keychain
          rm -f cert.p12 DeveloperIDG2CA.cer

      - name: Stage notarization API key (macOS only)
        if: runner.os == 'macOS'
        run: |
          printf '%s' "${{ secrets.APPLE_API_KEY_P8 }}" > "$RUNNER_TEMP/apple_api_key.p8"
          echo "APPLE_API_KEY=$RUNNER_TEMP/apple_api_key.p8" >> "$GITHUB_ENV"
```

Then add these to the **"Build + publish desktop app"** step's existing `env:`
block (electron-builder reads them for notarization — `APPLE_API_KEY` must be a
**file path**, which the step above sets):

```yaml
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          APPLE_API_KEY: ${{ env.APPLE_API_KEY }}          # path to the .p8
          APPLE_API_KEY_ID: ${{ secrets.APPLE_API_KEY_ID }}
          APPLE_API_ISSUER: ${{ secrets.APPLE_API_ISSUER_ID }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
```

> On the OTHER two OS legs (Windows/Linux) these env vars are simply absent, so
> nothing changes there. The `if: runner.os == 'macOS'` steps no-op off-mac.

## Step 7 — Verify (on any Mac, after a release build)

Download the published `.dmg`, then:

```bash
spctl -a -vvv -t install SpyDE-*.dmg
#   → "accepted, source=Notarized Developer ID"
codesign -dv --verbose=4 /Applications/SpyDE.app
#   → shows Authority=Developer ID Application: … (TEAMID) + "runtime" flag
stapler validate SpyDE-*.dmg
#   → "The validate action worked!"
```

Do the same for the `.zip` — since it's the auto-update payload, it must be
notarized too, or updates re-trigger Gatekeeper.

---

## SpyDE-specific gotchas

- **The Python sidecar is the risky part.** Notarization scans the **bundled**
  payload (`uv`, the project source under `extraResources/python`), NOT the venv
  that's built on the user's first launch. If Apple rejects the submission, get
  the exact offending file with:
  ```bash
  xcrun notarytool log <submission-id> --key <p8> --key-id <id> --issuer <uuid>
  ```
  A bundled binary that isn't signed/hardened is the usual culprit.
- **First-run venv is fine.** Gatekeeper gates the *initial app launch*; once the
  app is trusted, spawning Python + loading its wheels later is not re-checked
  (that's why `disable-library-validation` + notarizing the app is enough — the
  first-run-downloaded wheels don't need their own notarization).
- **Both dmg AND zip are signed+notarized.** electron-builder signs the `.app`
  once, then packages both. The zip matters because it's what electron-updater
  applies — notarizing it is what makes silent auto-updates trusted.
- **Time/cost.** Notarization adds ~2–10 min to the mac leg (Apple's queue).
  Budget for it in release timing.
- **Windows** is a separate track — an EV/OV code-signing cert (~$200–400/yr) +
  `win.certificateFile`/`signtool`. Not covered here; SmartScreen reputation also
  builds over download volume. See `DISTRIBUTION_PLAN.md §5`.

## Rollback

Signing is opt-in via config: if a release must ship before certs are sorted,
revert the `mac:` block to `identity: null` (and drop the `hardenedRuntime` /
`notarize` keys) — the build goes back to unsigned. Nothing else depends on it.
