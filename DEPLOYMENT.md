# Deployment Guide: GitHub + Railway

## Step 1: Push to GitHub

### If you don't have a GitHub repo yet:

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name it (e.g., `car-auction-prediction`)
   - Choose Public or Private
   - **Don't** initialize with README (we already have one)
   - Click "Create repository"

2. **Initialize git and push:**
   ```bash
   cd "/Users/nikitarogers/Anton Project"
   git init
   git add .
   git commit -m "Initial commit: Car auction price prediction pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repo name.

### If you already have a GitHub repo:

```bash
cd "/Users/nikitarogers/Anton Project"
git init
git add .
git commit -m "Initial commit: Car auction price prediction pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Important:** Your `.gitignore` ensures secrets and large files aren't committed. Never commit `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `.env` files.

---

## Step 2: Deploy to Railway

1. **Sign up / Log in:**
   - Go to https://railway.app
   - Sign up with GitHub (easiest)

2. **Create a new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository (`car-auction-prediction` or whatever you named it)
   - Railway will detect it's a Python app (sees `requirements.txt` and `Procfile`)

3. **Set environment variable(s):**
   - In Railway dashboard, go to your project → Variables
   - Add **OPENAI_API_KEY** (for OpenAI) and/or **ANTHROPIC_API_KEY** (for Claude). At least one is required; add both to compare providers in the web app.
   - Railway will restart the app automatically

4. **Optional: Persistent volume for regression models**
   - Add a volume in Railway (Command Palette → Add Volume)
   - Set mount path to `/data` (or set `RAILWAY_VOLUME_MOUNT_PATH` in Variables)
   - Training data and models persist across deploys; app auto-retrains on startup if CSV exists but pkl files do not

5. **Deploy:**
   - Railway auto-deploys when you push to GitHub
   - Or click "Deploy" in the dashboard
   - Wait ~2-3 minutes for build

6. **Get your URL:**
   - Railway gives you a URL like `https://your-app-name.up.railway.app`
   - Share this with your professor!

---

## Step 3: Test the Deployment

1. Visit your Railway URL (e.g., `https://your-app-name.up.railway.app`)
2. You should see the upload form
3. Click "Download Excel Template" or visit `/create_template` to get a sample file
4. Upload the template and run predictions
5. Verify results table appears and download works

---

## Troubleshooting

**"OPENAI_API_KEY is required" error:**
- Check Railway Variables: make sure `OPENAI_API_KEY` is set correctly
- No quotes needed in Railway (just paste the key value)

**Build fails:**
- Check Railway logs: click on your deployment → View Logs
- Common issues: missing dependencies in `requirements.txt`, Python version mismatch

**App doesn't start:**
- Check `Procfile` exists and has correct format
- Check Railway logs for gunicorn errors

**Can't push to GitHub:**
- Make sure you're authenticated: `gh auth login` or use SSH keys
- Check `.gitignore` isn't blocking important files

**"Failed to fetch" or WORKER TIMEOUT when uploading training CSV:**
- `gunicorn.conf.py` sets `timeout = 300`. Procfile and `railway.toml` use it via `gunicorn -c gunicorn.conf.py app:app`
- If it persists: Railway → your service → Settings → **Custom Start Command**. Remove it (so Procfile/railway.toml is used), or set it to exactly: `gunicorn -c gunicorn.conf.py app:app`
- Fallback: Add **Variable** `GUNICORN_CMD_ARGS` = `--timeout 300` so Gunicorn gets the timeout even if the start command omits it

---

## For Your Professor

Share this with them:

1. **Web app URL:** `https://your-app-name.up.railway.app`
2. **How to use:**
   - Visit the URL
   - Download template Excel (or use your own with columns: vehicle_id, make, model, year, mileage, price optional)
   - Upload Excel file
   - Select condition (P1–P4)
   - Click "Run Predictions"
   - View results table and download results as Excel

3. **Source code:** GitHub repo URL (if you want to share)

That's it! No installation needed for them.
