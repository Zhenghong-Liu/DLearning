<template>
	<div id="app">
		<h1>Othello AlphaZero</h1>

		<GameControls :current-player="currentPlayer" :board-size="boardSize" :game-status="gameStatus"
			:pass-action="hasPassAction" :piece-counts="pieceCounts" :is-processing="isProcessing"
			@new-game="startNewGame" @pass-turn="handleHumanMove({action: 'pass'})" />

		<OthelloBoard :board="boardState" :board-size="boardSize" :legal-moves="legalMoves" :last-move="lastMove"
			@move-made="handleHumanMove" />

		<div v-if="gameStatus !== 'Ongoing'" class="game-over-message">
			<h2>{{ gameStatus }}</h2>
			<button @click="startNewGame({ size: boardSize, firstPlayer: 1 })">Play Again (Human First)</button>
		</div>
	</div>
</template>

<script>
	import axios from 'axios';
	import OthelloBoard from './components/OthelloBoard.vue';
	import GameControls from './components/GameControls.vue';

	// 后端 API 地址，请确保你的 Flask/Python 后端运行在 5000 端口
	const API_BASE_URL = 'http://localhost:5001/api/game';

	// src/App.vue (在 <script> 内部，export default 外部)
	const FLIP_ANIMATION_DURATION = 0.4;
	// const AI_THINK_TIME = 0; // AI 的思考时间现在由后端控制，这里可以设为 0
	function waitForDelay(durationInSeconds) {
		return new Promise(resolve => setTimeout(resolve, durationInSeconds * 1000));
	}

	export default {
		name: 'App',
		components: {
			OthelloBoard,
			GameControls
		},
		data() {
			return {
				boardSize: 8, // 当前板子尺寸
				boardState: [], // 2D 数组
				currentPlayer: 1, // 1: Human (O), -1: AI (X)
				legalMoves: [], // 合法移动列表 [{x, y}, ...]
				lastMove: null, // 上一步移动 {x, y}
				gameStatus: 'Initializing',
				// pieceCounts: { white: 0, black: 0, empty: 0 }, // 黑白棋子计数
				isProcessing: false, // 防止重复点击和控制 AI 思考状态
			};
		},
		computed: {
			hasPassAction() {
				// 只有在游戏进行中且轮到人下，且没有合法移动时，才允许 Pass
				return this.legalMoves.length === 0 && this.gameStatus === 'Ongoing' && this.currentPlayer === 1;
			},
			pieceCounts() { // 【新增 computed 属性】
				let white = 0;
				let black = 0;

				// 遍历 boardState (一个 2D 数组)
				if (this.boardState && this.boardState.length > 0) {
					this.boardState.forEach(row => {
						row.forEach(piece => {
							if (piece === 1) { // 1 代表白棋 (Human)
								white++;
							} else if (piece === -1) { // -1 代表黑棋 (AI)
								black++;
							}
						});
					});
				}

				const totalSquares = this.boardSize * this.boardSize;
				const empty = totalSquares - white - black;

				return {
					white,
					black,
					empty
				};
			}
		},
		mounted() {
			// 应用加载时默认开始 8x8，人先手的游戏
			this.startNewGame({
				size: this.boardSize,
				firstPlayer: 1
			});
		},
		methods: {
			async startNewGame(config) {
				if (this.isProcessing) return;
				this.isProcessing = true;

				const {
					size,
					firstPlayer
				} = config;

				this.gameStatus = `Starting ${size}x${size} game...`;
				this.boardSize = size;

				try {
					const response = await axios.post(`${API_BASE_URL}/new`, {
						size: size,
						first_player: firstPlayer
					});

					this.updateGameState(response.data);
				} catch (error) {
					console.error('Error starting new game:', error.response ? error.response.data : error);
					this.gameStatus = 'Error: Cannot connect to Python Backend (check terminal)';
				} finally {
					this.isProcessing = false;
				}
			},
			// src/App.vue (在 methods 内部)

			async handleAIMove() {
				if (this.gameStatus !== 'Ongoing' || this.currentPlayer !== -1) return;

				this.isProcessing = true;
				this.gameStatus = 'AI is thinking...';

				try {
					// AI 的思考延迟已在后端 app.py 中实现
					const response = await axios.post(`${API_BASE_URL}/ai_move`);

					// 步骤 1: 更新棋盘状态（触发 AI 翻转动画）
					this.updateGameState(response.data);

					// 步骤 2: 等待 AI 翻转动画完成
					await waitForDelay(FLIP_ANIMATION_DURATION);

				} catch (error) {
					console.error('Error handling AI move:', error.response ? error.response.data : error);
					this.gameStatus = 'Error during AI move';
				} finally {
					this.isProcessing = false;

					// 【关键】检查 AI 下完后，游戏是否结束或轮到人
					if (this.gameStatus === 'Ongoing' && this.currentPlayer === -1) {
						// 如果 AI 下完后发现自己需要 Pass 或还有机会下，则再次触发 AI 移动
						// 确保连续 Pass 时的处理
						this.handleAIMove();
					}
				}
			},

			// src/App.vue (在 methods 内部)

			async handleHumanMove(coords) {
				// 【保留】前置校验和锁定
				if (this.gameStatus !== 'Ongoing' || this.currentPlayer !== 1 || this.isProcessing) return;

				const {
					x,
					y,
					action
				} = coords;

				// ... (合法性检查和 isProcessing = true 保持不变) ...

				this.isProcessing = true;
				this.gameStatus = 'Processing Move...';

				try {
					// 步骤 1: 发送人类移动请求 (后端只处理人类移动)
					const response = await axios.post(`${API_BASE_URL}/human_move`, {
						x,
						y,
						action
					});

					// 步骤 2: 更新棋盘状态 (触发人类翻转动画)
					this.updateGameState(response.data);

					// 步骤 3: 【关键】等待人类翻转动画完成
					await waitForDelay(FLIP_ANIMATION_DURATION);

					// 步骤 4: 检查游戏状态，如果轮到 AI，则触发 AI 移动
					if (this.gameStatus === 'Ongoing' && this.currentPlayer === -1) {

						// 立即调用 AI 移动函数，由 handleAIMove 接管后续流程和锁定状态
						this.handleAIMove();
						// 注意：这里 handleHumanMove 立即结束，isProcessing 会在 handleAIMove 中解除。

					} else {
						// 步骤 5: 如果游戏结束或人 Pass 后仍然是人下，解除锁定
						this.isProcessing = false;
					}

				} catch (error) {
					console.error('Error handling human move:', error.response ? error.response.data : error);
					this.gameStatus = 'Error during human move';
					this.isProcessing = false;
				}
				// 【移除】这里的 finally 块被移除，因为 isProcessing 的解除被转移到了 handleAIMove 或 else 分支中。
			},

			updateGameState(data) {
				this.boardState = data.board;
				this.legalMoves = data.legal_moves;
				this.currentPlayer = data.current_player;
				this.lastMove = data.last_move;
				this.gameStatus = data.status;
				// this.pieceCounts = data.piece_counts;

				if (this.gameStatus === 'Ongoing' && this.currentPlayer === -1 && !this.isProcessing) {
					// 如果轮到 AI 并且游戏仍在进行中，更新状态提示
					this.gameStatus = 'AI is thinking...';
				}
			}
		}
	};
</script>

<style>
	#app {
		font-family: Avenir, Helvetica, Arial, sans-serif;
		text-align: center;
		color: #2c3e50;
		margin-top: 20px;
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.game-over-message {
		margin-top: 20px;
		padding: 15px;
		border: 2px solid red;
		background-color: #ffe0e0;
	}
</style>