// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const fileUpload = document.getElementById('file-upload');
    const filePreview = document.getElementById('file-preview');
    const newChatBtn = document.getElementById('new-chat');
    const themeToggle = document.getElementById('theme-toggle');
    
    // Markdown converter
    const converter = new showdown.Converter({
        tables: true,
        simplifiedAutoLink: true,
        strikethrough: true,
        tasklists: true
    });
    
    // Dark mode toggle functionality
    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }
    
    // Check for saved theme preference or respect OS preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // Check if user prefers dark mode
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setTheme('dark');
        }
    }
    
    // Toggle theme
    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
    });
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Get file icon based on file type
    function getFileIcon(fileName) {
        const extension = fileName.split('.').pop().toLowerCase();
        
        // Map file extensions to Font Awesome icons
        const iconMap = {
            // Documents
            'pdf': 'fa-file-pdf',
            'txt': 'fa-file-alt',
            
            // Data
            'csv': 'fa-file-csv',
            'json': 'fa-file-code',
            
            // Archives
            'zip': 'fa-file-archive',
            
            // Code
            'py': 'fa-file-code',
            'js': 'fa-file-code',
            'html': 'fa-file-code',
            'css': 'fa-file-code',
            
            // Images
            'jpg': 'fa-image',
            'jpeg': 'fa-image',
            'png': 'fa-image',
            'gif': 'fa-image',
            'bmp': 'fa-image',
            'tiff': 'fa-image',
            
            // Audio
            'mp3': 'fa-music',
            'wav': 'fa-music',
            'ogg': 'fa-music',
            'flac': 'fa-music',
            
            // Video
            'mp4': 'fa-video',
            'avi': 'fa-video',
            'mov': 'fa-video',
            'mkv': 'fa-video'
        };
        
        const iconClass = iconMap[extension] || 'fa-file';
        return `<i class="fas ${iconClass}"></i>`;
    }
    
    // Display selected file with appropriate icon
    fileUpload.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            const fileSize = (file.size / 1024).toFixed(2);
            const fileIcon = getFileIcon(file.name);
            
            filePreview.innerHTML = `
                <div class="file-badge">
                    ${fileIcon}
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">(${fileSize} KB)</span>
                    <span class="remove-file">&times;</span>
                </div>
            `;
            
            // Add click event to remove file button
            document.querySelector('.remove-file').addEventListener('click', function(e) {
                e.stopPropagation();
                fileUpload.value = '';
                filePreview.innerHTML = '';
            });
            
            // Show file type specific message
            const fileExt = file.name.split('.').pop().toLowerCase();
            const processingMessages = {
                'pdf': 'PDF will be analyzed for text content',
                'csv': 'CSV data will be parsed and analyzed',
                'json': 'JSON structure will be parsed',
                'jpg': 'Image will be processed with OCR to extract text',
                'jpeg': 'Image will be processed with OCR to extract text',
                'png': 'Image will be processed with OCR to extract text',
                'mp3': 'Audio will be transcribed to text',
                'wav': 'Audio will be transcribed to text',
                'mp4': 'Video will be analyzed for frames and audio',
                'zip': 'ZIP archive will be extracted and contents analyzed'
            };
            
            if (processingMessages[fileExt]) {
                const processingInfo = document.createElement('div');
                processingInfo.className = 'processing-info';
                processingInfo.innerHTML = `<i class="fas fa-info-circle"></i> ${processingMessages[fileExt]}`;
                filePreview.appendChild(processingInfo);
            }
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
                            ${getFileIcon(fileUpload.files[0].name)}
                            <span>${fileUpload.files[0].name}</span>
                        </div>
                    ` : ''}
                </div>
            `;
        } else if (sender === 'system') {
            messageContent = `
                <div class="message-content system-message">
                    <p>${message}</p>
                </div>
            `;
            messageDiv.classList.add('system');
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
    
        console.log('[chat] submit handler fired');
        const message = userInput.value.trim();
        const hasFile = fileUpload.files.length > 0;
    
        // guard: must have a message or a file
        if (!message && !hasFile) {
            console.log('[chat] nothing to send, aborting');
            return;
        }
    
        // show user's own message (if any)
        if (message) addMessageToChat(message, 'user');
    
        // build formData
        const formData = new FormData();
        formData.append('message', message);
        if (hasFile) {
            const file = fileUpload.files[0];
            console.log(`[chat] attaching file ${file.name}`, file);
            formData.append('file', file);
        }
    
        // Create thinking animation with dots
        const thinkingDiv = document.createElement('div');
        thinkingDiv.classList.add('message', 'bot-message', 'thinking-message');
        thinkingDiv.innerHTML = `
            <div class="message-content">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(thinkingDiv);
        scrollToBottom();
    
        try {
            console.log('[chat] sending POST /chat');
            const res = await fetch('/chat', {
                method: 'POST',
                body: formData
            });
            console.log('[chat] got response', res.status);
    
            const data = await res.json();
    
            // remove thinking
            thinkingDiv.remove();
            // remove any lingering system msgs
            document.querySelectorAll('.system').forEach(el => el.remove());
    
            addMessageToChat(data.response, 'bot');
    
        } catch (err) {
            console.error('[chat] fetch error', err);
            thinkingDiv.remove();
            addMessageToChat('Error sending request. See console.', 'bot');
        } finally {
            // *only now* clear inputs
            userInput.value = '';
            fileUpload.value = '';
            filePreview.innerHTML = '';
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
                        <div class="file-type"><i class="fas fa-image"></i> Images</div>
                        <div class="file-type"><i class="fas fa-music"></i> Audio</div>
                        <div class="file-type"><i class="fas fa-video"></i> Video</div>
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error clearing chat history:', error);
            addMessageToChat('Failed to clear chat history. Please refresh the page.', 'system');
        }
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter or Cmd+Enter to submit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            chatForm.dispatchEvent(new Event('submit'));
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            userInput.value = '';
            fileUpload.value = '';
            filePreview.innerHTML = '';
            userInput.focus();
        }
        
        // Ctrl+N or Cmd+N for new chat
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault(); // Prevent new browser window
            newChatBtn.click();
        }
        
        // Ctrl+U or Cmd+U to open file upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            fileUpload.click();
        }
    });
    
    // Focus input on page load
    userInput.focus();
    
    // Auto-resize input field as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        const maxHeight = 150; // Maximum height before scrolling
        this.style.height = Math.min(this.scrollHeight, maxHeight) + 'px';
    });
    
    // Paste image from clipboard
    document.addEventListener('paste', function(e) {
        const clipboardData = e.clipboardData || window.clipboardData;
        const items = clipboardData.items;
        
        if (!items) return;
        
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const fileList = new DataTransfer();
                fileList.items.add(blob);
                
                // Set file to input and trigger change event
                fileUpload.files = fileList.files;
                fileUpload.dispatchEvent(new Event('change'));
                
                e.preventDefault();
                break;
            }
        }
    });
    
    // Add active animation for send button
    const sendBtn = document.getElementById('send-btn');
    sendBtn.addEventListener('mousedown', function() {
        this.classList.add('sending');
    });
    
    sendBtn.addEventListener('mouseup', function() {
        this.classList.remove('sending');
    });
    
    // Add file drop area
    const dropArea = document.createElement('div');
    dropArea.className = 'file-drop-area';
    dropArea.innerHTML = `
        <div class="drop-message">
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drop file here to upload</p>
        </div>
    `;
    document.body.appendChild(dropArea);
    
    // File drag and drop handlers
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropArea.classList.add('active');
    });
    
    document.addEventListener('dragleave', function(e) {
        const rect = dropArea.getBoundingClientRect();
        if (
            e.clientX < rect.left ||
            e.clientX >= rect.right ||
            e.clientY < rect.top ||
            e.clientY >= rect.bottom
        ) {
            dropArea.classList.remove('active');
        }
    });
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
        dropArea.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            fileUpload.files = e.dataTransfer.files;
            fileUpload.dispatchEvent(new Event('change'));
        }
    });
    
    // Handle initial messages if any
    document.querySelectorAll('.markdown-content pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
    
    // Scroll to bottom on initial load
    scrollToBottom();
});