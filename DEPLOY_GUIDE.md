# Beginner guide: put your project online for free

You do not need Git or any terminal commands. Everything is done in the web browser.

## Part 1 - Get the files ready
1. Download **AKTU_Placement_Project.zip** from the chat.
2. Find it in your *Downloads* folder.
3. **Windows:** right-click the ZIP → **Extract All…** → **Extract**. **Mac:** double-click the ZIP.
4. Open the extracted folder `AKTU_Placement_Project`. You should see `app.py`, `analysis.py`, `plots.py`,
   `requirements.txt`, `placement_data_filled.csv` and more. Keep this window open.

## Part 2 - Create a GitHub account
1. Go to **github.com** and click **Sign up**.
2. Enter your email, a password and a username (this username becomes part of your project's link). Click **Continue**.
3. Solve the puzzle, then enter the code GitHub emails you.
4. Choose the free plan if asked.

## Part 3 - Create a repository (a folder on GitHub)
1. After signing in, click the **+** icon at the top right → **New repository**.
2. **Repository name:** `aktu-placement-analytics` (small letters, no spaces).
3. Select **Public** (Streamlit's free hosting needs a public repository).
4. Leave "Add a README file" **unchecked** (your project already has one).
5. Click **Create repository**.

## Part 4 - Upload the files
1. On the new empty repository page, click the link **uploading an existing file**.
2. Open the extracted folder, press **Ctrl + A** (Mac: **Cmd + A**) to select everything inside it, and drag it into
   the GitHub page. Drag the files and folders *inside* `AKTU_Placement_Project`, not the outer folder itself.
3. Wait until every file shows as uploaded. Check that `app.py`, `analysis.py`, `plots.py`, `requirements.txt` and
   `placement_data_filled.csv` are in the list. Folders `.streamlit` and `tests` should appear too.
   * Mac: files starting with a dot are hidden. Press **Cmd + Shift + .** in the folder to show them.
   * If the `.streamlit` folder will not upload, the app still works (only the theme colours change).
4. Scroll down and click the green **Commit changes** button.
5. You should now see `app.py` listed directly on the repository page (not inside another folder).

## Part 5 - Connect Streamlit Community Cloud
1. Go to **share.streamlit.io**.
2. Click **Continue with GitHub** (or Sign in), enter your GitHub details and click **Authorize**.
3. If asked for details, fill in your name and email and continue.

## Part 6 - Deploy
1. Click **Create app** (top right).
2. When asked "Do you already have an app?", click **Yup, I have an app**.
3. **Repository:** choose `your-username/aktu-placement-analytics` (type it if it is not suggested).
4. **Branch:** `main`.
5. **Main file path:** `app.py`.
6. **App URL (optional):** type a short name such as `aktu-placement` if you want a neat link.
7. Click **Advanced settings**, choose a recent **Python version** (3.12 or newer), click **Save**.
8. Click **Deploy**. Wait a few minutes while it installs the libraries ("Your app is in the oven").

## Part 7 - Your public link
When it finishes, your dashboard opens and the address in the browser bar (ending in `.streamlit.app`) is your
public website link. Copy it and share it. You can also find it any time on share.streamlit.io in your app list.
Free apps can go to sleep when nobody uses them for a while; open the link and click the button to wake it up.

## Part 8 - Update the website later
**Small change:** on GitHub open the file → click the **pencil icon** (Edit) → make the change → click
**Commit changes**. The website updates automatically in a minute or two.
**Replace a file (for example a new CSV):** open the repository → **Add file → Upload files** → drag the new file
with the same name → **Commit changes**.
If the site does not refresh: on share.streamlit.io click the **⋮** menu next to your app → **Reboot app**.

## If something goes wrong
* **"ModuleNotFoundError"** - `requirements.txt` is missing or not in the top level of the repository.
* **"FileNotFoundError ... placement_data_filled.csv"** - the CSV was not uploaded to the same place as `app.py`.
* **Build error** - on the app page click **Manage app** and read the log; try another Python version in
  Advanced settings.
