1. First verify your remote connection is correct
   git remote -v # Should show your GitHub repository URL
   If it shows nothing or wrong URL, then set it up with:
   git remote add origin https://github.com/victor-explore/victor-explore.git

2. Add all changed files to staging
   git add . # Adds all modified files to the staging area

3. Commit changes with a message
   git commit -m "your commit message" # Commits the staged changes with a message

4. Push changes to GitHub
   git push -u origin main # Use this for first push of the session
   For subsequent pushes, just use:
   git push # Pushes the committed changes to the remote repository
   git push -f origin main # Use with caution! This will overwrite remote changes
