// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const fileUpload = document.getElementById('file-upload');
    const filePreview = document.getElementById('file-preview');
    const newChatBtn = document.getElementById('new-chat');
    
    // Markdown converter
    const converter = new showdown.Converter({
        tables: true,
        simplifiedAutoLink: true,
        strikethrough: true,
        tasklists: true
    });
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Display selected file
    fileUpload.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            filePreview.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="remove-file">&times;</span>
            `;
            
            // Add click event to remove file button
            document.querySelector('.remove-file').addEventListener('click', function() {
                fileUpload.value = '';
                filePreview.innerHTML = '';
            });
        }
    });
    
    // Add message to chat
    function addMessageToChat(message, sender) {
        // Clear welcome message if present
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        let messageContent = '';
        if (sender === 'user') {
            messageContent = `
                <div class="message-content">
                    <p>${message}</p>
                    ${fileUpload.files.length > 0 ? `
                        <div class="file-indicator">
                            <i class="fas fa-paperclip"></i> ${fileUpload.files[0].name}
                        </div>
                    ` : ''}
                </div>
            `;
        } else {
            // Convert markdown to HTML for bot messages
            const htmlContent = converter.makeHtml(message);
            messageContent = `
                <div class="message-content">
                    <div class="markdown-content">${htmlContent}</div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageContent;
        chatContainer.appendChild(messageDiv);
        
        // Apply syntax highlighting to code blocks
        if (sender === 'bot') {
            messageDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
        
        scrollToBottom();
    }
    
    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message && fileUpload.files.length === 0) return;
        
        // Add user message to chat
        if (message) {
            addMessageToChat(message, 'user');
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('message', message);
        
        if (fileUpload.files.length > 0) {
            formData.append('file', fileUpload.files[0]);
        }
        
        // Show thinking animation
        const thinkingDiv = document.createElement('div');
        thinkingDiv.classList.add('message', 'bot-message');
        thinkingDiv.innerHTML = `
            <div class="message-content">
                <div class="thinking">
                    <span>Thinking</span>
                    <span class="dot">.</span>
                    <span class="dot">.</span>
                    <span class="dot">.</span>
                </div>
            </div>
        `;
        chatContainer.appendChild(thinkingDiv);
        scrollToBottom();
        
        // Clear input and file
        userInput.value = '';
        fileUpload.value = '';
        filePreview.innerHTML = '';
        
        try {
            // Send request to server
            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Remove thinking animation
            chatContainer.removeChild(thinkingDiv);
            
            // Add bot response to chat
            addMessageToChat(data.response, 'bot');
            
        } catch (error) {
            console.error('Error:', error);
            chatContainer.removeChild(thinkingDiv);
            addMessageToChat('Sorry, there was an error processing your request. Please try again.', 'bot');
        }
    });
    
    // New chat button
    newChatBtn.addEventListener('click', async function() {
        try {
            await fetch('/clear', {
                method: 'POST'
            });
            
            // Clear chat and add welcome message
            chatContainer.innerHTML = `
                <div class="welcome-message">
                    <h2>Welcome to Gemini AI Assistant</h2>
                    <p>Upload files or ask questions to get started</p>
                    <div class="file-type-icons">
                        <div class="file-type"><i class="fas fa-file-pdf"></i> PDF</div>
                        <div class="file-type"><i class="fas fa-file-archive"></i> ZIP</div>
                        <div class="file-type"><i class="fas fa-file-csv"></i> CSV</div>
                        <div class="file-type"><i class="fas fa-file-code"></i> JSON</div>
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error clearing chat history:', error);
        }
    });
    
    // Add animated dots effect to thinking animation
    setInterval(() => {
        const dots = document.querySelectorAll('.thinking .dot');
        dots.forEach((dot, index) => {
            setTimeout(() => {
                dot.style.opacity = (dot.style.opacity === '0' ? '1' : '0');
            }, index * 200);
        });
    }, 500);

    // Additional CSS for thinking animation
    const style = document.createElement('style');
    style.textContent = `
        .thinking {
            display: flex;
            align-items: center;
        }
        .thinking .dot {
            margin-left: 3px;
            font-weight: bold;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        .thinking .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        .thinking .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
    `;
    document.head.appendChild(style);
});