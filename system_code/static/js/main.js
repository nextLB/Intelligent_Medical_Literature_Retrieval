document.addEventListener('DOMContentLoaded', function() {
    console.log('医学文献智能检索系统已加载');
});

function showLoading(element) {
    element.innerHTML = '<div class="loading">加载中...</div>';
}

function showError(element, message) {
    element.innerHTML = `<div class="error">${message}</div>`;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN');
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
