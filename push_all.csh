#!/bin/tcsh
# Enhanced script to push to all GitHub accounts automatically

echo "🚀 Pushing to all GitHub accounts..."
echo "===================================="
echo ""

# Show current branch
set current_branch = `git branch --show-current`
echo "📍 Current branch: $current_branch"
echo ""

echo "📊 Available remotes:"
git remote -v
echo ""

# Get list of all remotes
set remotes = (`git remote`)
set success_count = 0
set total_count = 0
set failed_remotes = ()

foreach remote ($remotes)
    echo "🔄 Pushing to $remote (https://github.com/*/repo)..."
    
    # Push current branch to remote
    git push $remote $current_branch
    set push_status = $status
    @ total_count++
    
    if ($push_status == 0) then
        echo "✅ Successfully pushed to $remote"
        @ success_count++
        
        # Show the GitHub URL for easy access
        set remote_url = `git remote get-url $remote`
        echo "   🌐 View at: $remote_url"
    else
        echo "❌ Failed to push to $remote"
        set failed_remotes = ($failed_remotes $remote)
        
        # Try to get more info about the failure
        echo "   ℹ️  This might be due to:"
        echo "      - Repository doesn't exist on GitHub"
        echo "      - No push permissions"
        echo "      - Authentication required"
    endif
    echo ""
end

echo "📈 SUMMARY"
echo "=========="
echo "✅ Successful: $success_count/$total_count remotes"

if ($success_count == $total_count) then
    echo "🎉 All GitHub accounts updated successfully!"
    echo ""
    echo "Your code is now available on:"
    foreach remote ($remotes)
        set remote_url = `git remote get-url $remote`
        echo "  • $remote_url"
    end
else
    echo "⚠️  Some pushes failed:"
    foreach failed_remote ($failed_remotes)
        echo "  ❌ $failed_remote"
    end
    echo ""
    echo "💡 To fix authentication issues, you may need to:"
    echo "   1. Set up a Personal Access Token on GitHub"
    echo "   2. Use: git config credential.helper store"
    echo "   3. Or set up SSH keys for passwordless access"
endif

echo ""
echo "🏁 Push operation completed!"
