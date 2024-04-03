const chatbot = document.getElementById('chatbot');
const toggleBtn = document.getElementById('toggle-btn');
const minimizeBtn = document.getElementById('minimize-btn');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatbox = document.getElementById('chatbox');

let isChatbotVisible = false;

toggleBtn.addEventListener('click', () => {
    if (!isChatbotVisible) {
        chatbot.style.display = 'block'; // Show chatbot
        isChatbotVisible = true;
        toggleBtn.style.display = 'none'; // Hide toggle button
    }
});

minimizeBtn.addEventListener('click', () => {
    chatbot.style.display = 'none'; // Hide chatbot
    toggleBtn.style.display = 'block'; // Show toggle button
    isChatbotVisible = false;
});


// Listen for Enter key press event in the input field
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

sendBtn.addEventListener('click', () => {
    sendMessage();
});

// Function to send user message and receive response
function sendMessage() {
    const userMessage = userInput.value.trim();
    if (userMessage !== '') {
        appendMessage('You: ' + userMessage, 'outgoing');
        userInput.value = ''; // Clear the input field
        userInput.classList.remove('error-input'); // Remove error class to reset border color

        // Clear error messages
        clearErrorMessages();

        // Send user message to server
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_input: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.response;
            appendMessage('Bot: ' + botResponse, 'incoming');
        })
        .catch(error => {
            console.error('Error:', error);
            appendMessage('Error occurred. Please try again.', 'error');
        });
    } else {
        userInput.classList.add('error-input'); // Add error class to set border color to red
        appendMessage('Bot: Please input your query first.', 'error'); // Display error message
    }
}

// Function to clear error messages
function clearErrorMessages() {
    const errorMessages = chatbox.querySelectorAll('.error');
    errorMessages.forEach(message => {
        message.remove();
    });
}

// Function to append a message to the chatbox
function appendMessage(message, type) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', type);
    messageElement.innerText = message;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
}