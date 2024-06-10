// Function to extract comments
function getYouTubeComments() {
    const comments = [];
    document.querySelectorAll('#content-text').forEach(comment => {
        comments.push(comment.innerText);
    });
    return comments;
}

// Scroll to load more comments and collect them
async function scrollAndCollectComments() {
    let lastHeight = 0;
    let currentHeight = document.documentElement.scrollHeight;
    let comments = [];

    while (lastHeight !== currentHeight) {
        lastHeight = currentHeight;
        window.scrollTo(0, currentHeight);
        
        // Wait for more comments to load
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        currentHeight = document.documentElement.scrollHeight;
        comments = getYouTubeComments();
        console.log(`Loaded ${comments.length} comments`);
    }

    return comments;
}

// Function to download comments as a UTF-8 encoded file
function downloadComments(comments) {
    const blob = new Blob([comments.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'comments.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Run the script and save comments to a file
scrollAndCollectComments().then(comments => {
    console.log('All comments loaded:');
    console.log(comments);
    downloadComments(comments);
});
