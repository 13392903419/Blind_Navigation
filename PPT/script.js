/**
 * 英特尔AI解决方案答辩PPT - 交互脚本
 * 苹果风格演示系统
 */

class PresentationController {
    constructor() {
        this.currentSlide = 1;
        this.totalSlides = 15;
        this.isAnimating = false;
        this.touchStartX = 0;
        this.touchEndX = 0;
        
        this.init();
    }

    init() {
        // 获取DOM元素
        this.slides = document.querySelectorAll('.slide');
        this.navDots = document.querySelectorAll('.nav-dot');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.progressFill = document.getElementById('progressFill');
        this.pageIndicator = document.getElementById('pageIndicator');
        
        // 绑定事件
        this.bindEvents();
        
        // 初始化显示
        this.updateUI();
        
        // 启动动画
        this.animateCurrentSlide();
        
        console.log('🎯 PPT演示系统已启动');
        console.log('💡 提示：使用方向键或空格键翻页，↑↓或鼠标滚轮滚动内容，按F进入全屏');
    }

    bindEvents() {
        // 导航按钮点击
        this.prevBtn.addEventListener('click', () => this.prevSlide());
        this.nextBtn.addEventListener('click', () => this.nextSlide());
        
        // 侧边导航点击
        this.navDots.forEach(dot => {
            dot.addEventListener('click', () => {
                const slideNum = parseInt(dot.dataset.slide);
                this.goToSlide(slideNum);
            });
        });

        // 键盘事件
        document.addEventListener('keydown', (e) => this.handleKeydown(e));

        // 触摸事件（移动端支持）
        document.addEventListener('touchstart', (e) => {
            this.touchStartX = e.changedTouches[0].screenX;
        });

        document.addEventListener('touchend', (e) => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
        });

        // 鼠标滚轮翻页已禁用 - 滚轮只用于页内滚动
        // 如果用户在可滚动区域内，滚轮滚动内容；否则不做任何事

        // 窗口大小变化
        window.addEventListener('resize', () => this.handleResize());
    }

    handleKeydown(e) {
        // 获取当前滑页的滚动容器
        const currentSlideEl = document.querySelector(`.slide[data-slide="${this.currentSlide}"]`);
        const scrollContainer = currentSlideEl?.querySelector('.slide-inner.scrollable');
        
        switch(e.key) {
            case 'ArrowRight':
            case ' ':
            case 'Enter':
            case 'PageDown':
                e.preventDefault();
                this.nextSlide();
                break;
            case 'ArrowLeft':
            case 'Backspace':
            case 'PageUp':
                e.preventDefault();
                this.prevSlide();
                break;
            case 'ArrowDown':
                // 如果有可滚动区域，向下滚动；否则下一页
                if (scrollContainer) {
                    const maxScroll = scrollContainer.scrollHeight - scrollContainer.clientHeight;
                    if (scrollContainer.scrollTop < maxScroll - 10) {
                        scrollContainer.scrollBy({ top: 100, behavior: 'smooth' });
                        e.preventDefault();
                        return;
                    }
                }
                e.preventDefault();
                this.nextSlide();
                break;
            case 'ArrowUp':
                // 如果有可滚动区域，向上滚动；否则上一页
                if (scrollContainer) {
                    if (scrollContainer.scrollTop > 10) {
                        scrollContainer.scrollBy({ top: -100, behavior: 'smooth' });
                        e.preventDefault();
                        return;
                    }
                }
                e.preventDefault();
                this.prevSlide();
                break;
            case 'Home':
                e.preventDefault();
                this.goToSlide(1);
                break;
            case 'End':
                e.preventDefault();
                this.goToSlide(this.totalSlides);
                break;
            case 'f':
            case 'F':
                e.preventDefault();
                this.toggleFullscreen();
                break;
            case 'Escape':
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                }
                break;
            // 数字键直接跳转 (0表示第10页)
            case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8':
            case '9': case '0':
                e.preventDefault();
                const num = e.key === '0' ? 10 : parseInt(e.key);
                if (num <= this.totalSlides) {
                    this.goToSlide(num);
                }
                break;
        }
    }

    handleSwipe() {
        const swipeThreshold = 50;
        const diff = this.touchStartX - this.touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                this.nextSlide();
            } else {
                this.prevSlide();
            }
        }
    }

    handleResize() {
        // 可以在这里添加响应式逻辑
    }

    prevSlide() {
        if (this.currentSlide > 1 && !this.isAnimating) {
            this.goToSlide(this.currentSlide - 1);
        }
    }

    nextSlide() {
        if (this.currentSlide < this.totalSlides && !this.isAnimating) {
            this.goToSlide(this.currentSlide + 1);
        }
    }

    goToSlide(slideNum) {
        if (slideNum === this.currentSlide || this.isAnimating) return;
        if (slideNum < 1 || slideNum > this.totalSlides) return;

        this.isAnimating = true;

        // 获取当前和目标幻灯片
        const currentSlideEl = document.querySelector(`.slide[data-slide="${this.currentSlide}"]`);
        const targetSlideEl = document.querySelector(`.slide[data-slide="${slideNum}"]`);

        // 重置目标页滚动位置
        const targetScrollContainer = targetSlideEl.querySelector('.slide-inner.scrollable');
        if (targetScrollContainer) {
            targetScrollContainer.scrollTop = 0;
        }

        // 移除当前幻灯片的激活状态
        currentSlideEl.classList.remove('active');
        currentSlideEl.classList.add('exit');

        // 设置目标幻灯片方向
        if (slideNum > this.currentSlide) {
            targetSlideEl.style.transform = 'translateX(100px)';
        } else {
            targetSlideEl.style.transform = 'translateX(-100px)';
        }

        // 激活目标幻灯片
        setTimeout(() => {
            currentSlideEl.classList.remove('exit');
            targetSlideEl.classList.add('active');
            targetSlideEl.style.transform = '';
        }, 50);

        // 更新当前页码
        this.currentSlide = slideNum;
        this.updateUI();

        // 动画完成
        setTimeout(() => {
            this.isAnimating = false;
            this.animateCurrentSlide();
        }, 600);
    }

    updateUI() {
        // 更新进度条
        const progress = (this.currentSlide / this.totalSlides) * 100;
        this.progressFill.style.width = `${progress}%`;

        // 更新页码
        this.pageIndicator.querySelector('.current-page').textContent = this.currentSlide;
        this.pageIndicator.querySelector('.total-pages').textContent = this.totalSlides;

        // 更新导航点
        this.navDots.forEach(dot => {
            dot.classList.toggle('active', parseInt(dot.dataset.slide) === this.currentSlide);
        });

        // 更新导航按钮状态
        this.prevBtn.disabled = this.currentSlide === 1;
        this.nextBtn.disabled = this.currentSlide === this.totalSlides;

        // 更新URL hash（方便分享特定页）
        history.replaceState(null, null, `#slide-${this.currentSlide}`);
    }

    animateCurrentSlide() {
        const currentSlideEl = document.querySelector(`.slide[data-slide="${this.currentSlide}"]`);
        
        // 重置动画
        const cards = currentSlideEl.querySelectorAll('.glass-card');
        cards.forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
        });

        // 触发动画
        setTimeout(() => {
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 80);
            });
        }, 100);

        // 特殊动画：统计条
        this.animateStatBars(currentSlideEl);
    }

    animateStatBars(slideEl) {
        const statBars = slideEl.querySelectorAll('.stat-fill');
        statBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0';
            setTimeout(() => {
                bar.style.transition = 'width 1s ease';
                bar.style.width = width;
            }, 500);
        });
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().then(() => {
                document.body.classList.add('fullscreen');
            }).catch(err => {
                console.log('全屏模式不可用:', err);
            });
        } else {
            document.exitFullscreen().then(() => {
                document.body.classList.remove('fullscreen');
            });
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    const presentation = new PresentationController();
    
    // 检查URL hash，跳转到指定页
    const hash = window.location.hash;
    if (hash && hash.startsWith('#slide-')) {
        const slideNum = parseInt(hash.replace('#slide-', ''));
        if (slideNum >= 1 && slideNum <= presentation.totalSlides) {
            setTimeout(() => {
                presentation.goToSlide(slideNum);
            }, 100);
        }
    }
});

// 监听全屏变化
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        document.body.classList.remove('fullscreen');
    }
});

// 添加打印支持
window.addEventListener('beforeprint', () => {
    document.querySelectorAll('.slide').forEach(slide => {
        slide.classList.add('active');
        slide.style.position = 'relative';
        slide.style.pageBreakAfter = 'always';
    });
});

window.addEventListener('afterprint', () => {
    document.querySelectorAll('.slide').forEach((slide, index) => {
        if (index !== 0) {
            slide.classList.remove('active');
        }
        slide.style.position = '';
        slide.style.pageBreakAfter = '';
    });
});

// 添加一些实用工具函数
const PPTUtils = {
    // 导出为PDF（提示用户使用打印功能）
    exportPDF() {
        alert('请使用浏览器的打印功能 (Ctrl+P) 并选择"保存为PDF"');
        window.print();
    },
    
    // 获取当前页码
    getCurrentSlide() {
        return document.querySelector('.slide.active').dataset.slide;
    },
    
    // 重置演示
    reset() {
        location.hash = '#slide-1';
        location.reload();
    }
};

// 暴露给全局，方便调试
window.PPTUtils = PPTUtils;

console.log('%c🎨 视障人士出行辅助系统 - 答辩PPT', 'color: #0071c5; font-size: 20px; font-weight: bold;');
console.log('%c基于计算机视觉与大型语言模型', 'color: #00aeef; font-size: 14px;');
console.log('%c使用 PPTUtils.exportPDF() 导出PDF', 'color: #666;');
