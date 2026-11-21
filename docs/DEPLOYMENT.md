# Deployment guide

The project has not been deployed to a cloud account by its author. This guide starts with the free options and is based on provider documentation checked in November 2026. Free tiers change, so confirm the current terms before relying on them.

## What can and cannot be free

| Part | Free option | Notes |
|---|---|---|
| Dashboard | Streamlit Community Cloud | Free, deploys from a GitHub repository. Apps may go to sleep when unused and have resource limits. |
| Daily job (download, features, signals) | Your own computer with Windows Task Scheduler | Free and the most reliable, because NSE accepts home connections. |
| Daily job in the cloud | GitHub Actions (free for standard runners in public repositories) or an Oracle Cloud Always Free VM | Risky, see the NSE warning below. |
| Model training | Your own computer | Several hours of CPU. Do it locally and keep the models on your machine. |

### NSE warning

NSE blocks many cloud and data center IP addresses. Forum reports show 403 errors from cloud hosts, and some users report the same from services outside India. Whether GitHub Actions or a particular Oracle region is accepted cannot be known in advance, so test it first with the manual workflow described below before building anything on top of it.

## Recommended free setup: local daily job plus hosted dashboard

The daily job runs on your PC, pushes only small report files to GitHub, and Streamlit Community Cloud shows them. No data or models leave your computer.

1. Create a GitHub repository (public is simplest for the free hosting. A private repository also works but needs extra permissions for Streamlit) and push the project. The `.gitignore` already excludes `data/`, `.venv/` and logs. The `reports/` folder is committed on purpose.
   ```powershell
   git init
   git add .
   git commit -m "Initial project"
   git branch -M main
   git remote add origin https://github.com/YOUR_NAME/niftyinsight.git
   git push -u origin main
   ```
   Do the first push by hand so Git stores your credentials for the scheduled job.
2. Open share.streamlit.io, sign in with GitHub, choose Create app, pick the repository, branch `main`, main file `dashboard/app.py`, and Python 3.11 under advanced settings. The file `dashboard/requirements.txt` keeps the install small.
3. Set the model mode once so the daily job matches the models you trained (use the mode you trained last):
   ```powershell
   setx ALGO_TARGET_MODE excess
   ```
   Open a new PowerShell window afterwards.
4. Test the daily script by hand:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_daily.ps1
   ```
   It runs the daily pipeline, commits `reports/` and pushes. The hosted dashboard shows the new files after the push.
5. Schedule it with Task Scheduler on weekday evenings, after NSE publishes the files (about 20:30 India time). Replace the path:
   ```powershell
   $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File C:\path\to\niftyinsight\scripts\run_daily.ps1"
   $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 8:30PM
   Register-ScheduledTask -TaskName "NSE Daily Signals" -Action $action -Trigger $trigger
   ```
   The computer must be on and online at that time. Enable "run task as soon as possible after a scheduled start is missed" in the task settings if it is sometimes off.

## Optional: test whether GitHub Actions can reach NSE

The file `.github/workflows/check-nse.yml` runs `check-sources` on GitHub runners when you start it by hand (Actions tab, then "check-nse", then Run workflow). If all three lines say `ok`, a scheduled cloud job is possible. If it fails with 403, use the local job above. Also note that scheduled workflows in a public repository can be paused by GitHub after a long period without repository activity.

Even if NSE is reachable, a cloud runner starts empty each time, so the daily job would need somewhere free to keep roughly 1.5 GB of raw files and the models between runs. That requires extra storage work that is not included in this project.

## Optional: full pipeline on an Oracle Cloud Always Free VM

Oracle's Always Free Ampere A1 allowance was reduced in June 2026 to 2 OCPUs and 12 GB of memory in total, according to news reports and Oracle's documentation. That is enough to run the daily job and the dashboard with Docker, but too slow for comfortable training. Account creation asks for a payment card for verification, and free capacity in a region is sometimes unavailable. Steps if you try it:

1. Create the VM, preferably in an India region, and install Docker.
2. Copy the project, copy your trained `data/models` and `data/processed` folders from your PC (training stays local), and run `docker compose up -d dashboard`.
3. Add a cron entry: `30 20 * * 1-5 cd /home/ubuntu/niftyinsight && docker compose run --rm pipeline`.
4. Restrict port 8501 to your own IP or put authentication in front of it.

## Before going live with real money

- Live orders: the `trade` command is a dry run unless `--live` is passed and `ENABLE_LIVE_TRADING=1` is set. The live path is untested against a real broker account.
- Secrets: broker credentials must come from environment variables or the platform secret store, never from files in the repository. Kite access tokens expire daily.
- The simulations in the README show no reliable profit. Do not trade real money on this system.
