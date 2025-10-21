<template>
	<div :class="['board-square', { 'is-legal': isLegalMove, 'last-move': isLastMove }]" @click="handleMove">
		<div v-if="piece !== 0" :class="['piece', pieceClass, { 'flip-animate': shouldAnimate }]"
			@animationend="onAnimationEnd"></div>

		<div v-if="isLegalMove" class="hint-dot"></div>
	</div>
</template>

<script>
	export default {
		name: 'BoardSquare',
		props: {
			piece: {
				type: Number,
				required: true
			}, // 1=White, -1=Black, 0=Empty
			isLegalMove: {
				type: Boolean,
				default: false
			},
			isLastMove: {
				type: Boolean,
				default: false
			},
			coords: {
				type: Object,
				required: true
			} // { x, y } coordinates
		},
		data() {
			return {
				shouldAnimate: false, // 【新增】控制动画的开关
			};
		},
		computed: {
			pieceClass() {
				return this.piece === 1 ? 'white-piece' : (this.piece === -1 ? 'black-piece' : '');
			}
		},
		watch: {
			piece(newVal, oldVal) {
				// 【新增 watch】
				// 只有在旧值和新值都不为 0 且颜色不同时，才触发动画
				if (oldVal !== 0 && newVal !== 0 && oldVal !== newVal) {
					this.shouldAnimate = true;
				}
			}
		},
		methods: {
			handleMove() {
				if (this.isLegalMove) {
					// 只有合法移动才触发事件
					this.$emit('make-move', this.coords);
				}
			},
			onAnimationEnd() {
				// 【新增方法】动画播放完毕后，移除类名，准备下一次动画
				this.shouldAnimate = false;
			}
		}
	};
</script>

<style scoped>
	.board-square {
		/* 使用 padding-bottom 实现自适应正方形 */
		width: 100%;
		padding-bottom: 100%;
		position: relative;
		background-color: #008000;
		/* 绿色板子 */
		border: 1px solid #333;
		box-sizing: border-box;
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.is-legal {
		cursor: pointer;
		background-color: #009900;
	}

	/* last-move 标记 */
	.last-move::after {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		border: 3px solid rgba(255, 255, 0, 0.8);
		/* 黄色标记 */
		pointer-events: none;
		z-index: 5;
	}

	.piece {
		position: absolute;
		width: 80%;
		height: 80%;
		border-radius: 50%;
		box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
		z-index: 10;
		top: 10%;
		/* transition: background-color 0.1s linear; */
		/* 【新增/修改】添加 background-color 过渡，使关键帧之间的颜色变化更柔和 */
		transition: background-color 0.8s ease-in-out;

		/* 【新增】确保 transform 始终应用于 piece 元素 */
		transform-origin: center;
	}

	.white-piece {
		background-color: white;
	}

	.black-piece {
		background-color: black;
	}

	.hint-dot {
		position: absolute;
		width: 20%;
		height: 20%;
		background-color: limegreen;
		border-radius: 50%;
		opacity: 0.8;
		z-index: 8;
		pointer-events: none;
		top: 40%
	}

	/* src/components/BoardSquare.vue (在 <style scoped> 中) */

	/* 优化后的关键帧动画：模拟翻转、颜色冲击和弹跳 */
	@keyframes flipImpact {

		/* 0%：初始状态 */
		0% {
			transform: scale(1.0) rotate(0deg) perspective(100px);
			opacity: 1;
			box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
		}

		/* 25%：收缩和颜色冲击点 (短暂显示过渡色) */
		25% {
			/* 沿 X 轴压扁，模拟翻转到侧面 */
			transform: scale(0.5, 1.2) rotate(30deg) perspective(100px);
			opacity: 0.7;
			/* 颜色冲击：快速过渡到一个中性色（例如灰色或亮色） */
			background-color: #A9A9A9;
		}

		/* 50%：最扁平点/颜色变化完成 (视觉上最模糊) */
		50% {
			transform: scale(0.1, 1.2) rotate(-30deg) perspective(100px);
			opacity: 0.9;
			/* 在这里背景颜色会被新的 pieceClass 覆盖，但动画效果仍在继续 */
			box-shadow: 0 0 10px rgba(255, 255, 0, 0.8);
			/* 闪光效果 */
		}

		/* 75%：反弹膨胀点 (新颜色，弹簧效果) */
		75% {
			transform: scale(1.2, 1.1) rotate(10deg) perspective(100px);
			opacity: 1;
			box-shadow: 0 5px 15px rgba(0, 0, 0, 0.6);
		}

		/* 100%：恢复最终状态 */
		100% {
			transform: scale(1.0) rotate(0deg) perspective(100px);
			opacity: 1;
			box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
		}
	}

	/* 触发动画的类 */
	.piece.flip-animate {
		/* 【修改时长】0.8 秒，以配合前端的节奏控制 */
		animation: flipImpact 0.8s ease-in-out;
	}
</style>
top: 10%