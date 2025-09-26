// content.js

// This function will be executed on the current webpage
(function() {
    console.log("AI Proctoring Script Injected!");

    // Create a video element to show the webcam feed
    const video = document.createElement('video');
    video.setAttribute('autoplay', 'true');
    video.style.position = 'fixed';
    video.style.bottom = '10px';
    video.style.right = '10px';
    video.style.width = '200px';
    video.style.border = '3px solid red';
    video.style.zIndex = '9999';
    document.body.appendChild(video);

    const canvas = document.createElement('canvas'); // Hidden canvas
    const context = canvas.getContext('2d');

    // Create a div to show alerts
    const alertBox = document.createElement('div');
    alertBox.style.position = 'fixed';
    alertBox.style.bottom = '10px';
    alertBox.style.left = '10px';
    alertBox.style.padding = '10px';
    alertBox.style.backgroundColor = 'white';
    alertBox.style.border = '2px solid black';
    alertBox.style.zIndex = '9999';
    alertBox.innerHTML = '<p id="status-text">Starting AI Proctor...</p>';
    document.body.appendChild(alertBox);
    const statusText = document.getElementById('status-text');

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => video.srcObject = stream)
        .catch(err => console.error("Webcam Error:", err));

    // Send frame to server every 2 seconds
    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        let imageData = canvas.toDataURL('image/jpeg', 0.8);

        // !! IMPORTANT: Use your Ngrok URL here !!
        const NGROK_URL = 'https://YOUR_UNIQUE_ID.ngrok.io/process_frame'; // Replace with your Ngrok URL

        fetch(NGROK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: imageData }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.warnings && data.warnings.length > 0) {
                statusText.style.color = 'red';
                statusText.innerHTML = data.warnings.join('<br>');
            } else {
                statusText.style.color = 'green';
                statusText.innerText = 'Status: OK';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusText.innerText = 'Cannot connect to AI server.';
        });

    }, 2000);
})();