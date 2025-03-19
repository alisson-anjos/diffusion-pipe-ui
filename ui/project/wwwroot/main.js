function openNewTab(url) {
    window.open(url, '_blank');
}

function scrollToBottom(element) {
    if (element) {
        element.scrollTop = element.scrollHeight;
    }
}

window.saveAsFile = (filename, base64) => {
    const link = document.createElement('a');
    link.download = filename;
    link.href = "data:application/octet-stream;base64," + base64;
    document.body.appendChild(link); // Necessário para o Firefox
    link.click();
    document.body.removeChild(link);
}

window.playAlarm = (soundUrl) => {
    const audioElement = document.getElementById("alarmAudio");
    if (audioElement) {
        audioElement.src = soundUrl;
        audioElement.play();
    }
};

window.stopAlarm = () => {
    const audioElement = document.getElementById("alarmAudio");
    if (audioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
    }
};