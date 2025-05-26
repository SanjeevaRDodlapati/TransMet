#!/bin/tcsh
# Super simple script for daily commits and pushes to both GitHub accounts

echo "🚀 Quick Commit & Push to All GitHub Accounts"
echo "=============================================="

# Check if we have uncommitted changes
set status_output = `git status --porcelain`
if ("$status_output" != "") then
    echo ""
    echo "📝 Uncommitted changes detected:"
    git status --short
    echo ""
    
    echo -n "💬 Enter commit message (or press Enter to see changes): "
    set commit_msg = "$<"
    
    if ("$commit_msg" == "") then
        echo ""
        echo "📋 Detailed changes:"
        git diff --stat
        echo ""
        echo -n "💬 Enter commit message: "
        set commit_msg = "$<"
    endif
    
    if ("$commit_msg" != "") then
        echo ""
        echo "📦 Adding all changes..."
        git add .
        
        echo "💾 Committing with message: '$commit_msg'"
        git commit -m "$commit_msg"
        
        if ($status == 0) then
            echo "✅ Commit successful!"
        else
            echo "❌ Commit failed!"
            exit 1
        endif
    else
        echo "❌ No commit message provided. Aborting."
        exit 1
    endif
else
    echo "ℹ️  No uncommitted changes detected."
endif

echo ""
echo "🌐 Pushing to all GitHub accounts..."
./push_all.csh
