fetch('/trainmodel', {
    method: 'POST'
})
    .then((response) => response.text())
    .then((body) => {
        if (body === 'Trained') {
            window.location.href = '/';
        }
    });
