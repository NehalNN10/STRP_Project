
// get comments of a page snapshot
function getYouTubeComments() {
    const comments = [];
    document.querySelectorAll('#content-text').forEach(comment => {
        comments.push(comment.innerText);
    });
    return comments;
}

// scroll down page to get comments
async function scrollAndCollectComments() {
    let lastHeight = 0;
    let currentHeight = document.documentElement.scrollHeight;
    let comments = [];

    while (lastHeight !== currentHeight) {
        lastHeight = currentHeight;
        window.scrollTo(0, currentHeight);
        
        // apparently this gives a delay for comments to load before replacing height
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        currentHeight = document.documentElement.scrollHeight;
        comments = getYouTubeComments();
        console.log(`Loaded ${comments.length} comments`);
    }

    return comments;
}

// download in UTF-8 because urdu text is involved
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

scrollAndCollectComments().then(comments => {
    console.log('All comments loaded:');
    console.log(comments);
    downloadComments(comments);
});
