// AI4K8s Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Add loading states to forms
    var forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function() {
            var submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            }
        });
    });

    // Add fade-in animation to cards
    var cards = document.querySelectorAll('.card');
    cards.forEach(function(card, index) {
        setTimeout(function() {
            card.classList.add('fade-in');
        }, index * 100);
    });
});

// Utility functions
function showNotification(message, type = 'info') {
    var alertClass = 'alert-' + type;
    var iconClass = type === 'success' ? 'fa-check-circle' : 
                   type === 'error' ? 'fa-exclamation-circle' : 
                   type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
    
    var notification = document.createElement('div');
    notification.className = 'alert ' + alertClass + ' alert-dismissible fade show position-fixed';
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas ${iconClass} me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(function() {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
    return Math.floor(seconds / 86400) + 'd ' + Math.floor((seconds % 86400) / 3600) + 'h';
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!', 'success');
    }).catch(function() {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

// Server management functions
function checkServerStatus(serverId) {
    fetch(`/api/server_status/${serverId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status) {
                showNotification('Server status updated', 'success');
                // Refresh the page to show updated status
                setTimeout(() => location.reload(), 1000);
            } else {
                showNotification('Failed to check server status', 'error');
            }
        })
        .catch(error => {
            console.error('Error checking server status:', error);
            showNotification('Error checking server status', 'error');
        });
}

function testServerConnection(serverId) {
    showNotification('Testing connection...', 'info');
    
    fetch(`/api/test_connection/${serverId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Connection test successful!', 'success');
        } else {
            showNotification('Connection test failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error testing connection:', error);
        showNotification('Error testing connection', 'error');
    });
}

// Chat functions
function sendQuickMessage(message) {
    var messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.value = message;
        messageInput.focus();
        
        // Trigger form submission
        var chatForm = document.getElementById('chat-form');
        if (chatForm) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
}

function clearChat() {
    var chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
        showNotification('Chat cleared', 'info');
    }
}

// Dashboard functions
function refreshDashboard() {
    showNotification('Refreshing dashboard...', 'info');
    location.reload();
}

function exportServerList() {
    // This would export the server list as CSV or JSON
    showNotification('Export functionality coming soon', 'info');
}

// Form validation
function validateForm(formId) {
    var form = document.getElementById(formId);
    if (!form) return false;
    
    var isValid = true;
    var requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(function(field) {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Auto-save functionality for forms
function enableAutoSave(formId, interval = 30000) {
    var form = document.getElementById(formId);
    if (!form) return;
    
    var formData = new FormData(form);
    var originalData = {};
    
    // Store original form data
    for (var pair of formData.entries()) {
        originalData[pair[0]] = pair[1];
    }
    
    setInterval(function() {
        var currentData = new FormData(form);
        var hasChanges = false;
        
        for (var pair of currentData.entries()) {
            if (originalData[pair[0]] !== pair[1]) {
                hasChanges = true;
                break;
            }
        }
        
        if (hasChanges) {
            // Auto-save logic would go here
            console.log('Form has unsaved changes');
        }
    }, interval);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for quick search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        var searchInput = document.querySelector('input[type="search"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        var modals = document.querySelectorAll('.modal.show');
        modals.forEach(function(modal) {
            var bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
    }
});

// Performance monitoring
function measurePerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                var perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
            }, 0);
        });
    }
}

// Initialize performance monitoring
measurePerformance();
