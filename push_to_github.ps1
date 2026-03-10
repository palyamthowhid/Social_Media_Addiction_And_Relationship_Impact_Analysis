# Initialize Git repository
git init

# Add all files to staging
git add .

# Commit the changes
git commit -m "Initial commit for ASIF application"

# Create main branch and add the remote repository you provided
git branch -M main
git remote add origin https://github.com/palyamthowhid/Social_Media_Addiction_And_Relationship_Impact_Analysis.git

# Push the code to GitHub
git push -u origin main

Write-Host "Success! Your code has been pushed to GitHub." -ForegroundColor Green
