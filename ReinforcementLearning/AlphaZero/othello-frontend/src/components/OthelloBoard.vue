<template>
  <div class="othello-board-container">
    <div class="coordinate-row">
      <div class="empty-corner"></div>
      <div v-for="c in boardSize" :key="'col-coord-' + c" class="coordinate-cell">{{ c - 1 }}</div>
    </div>

    <div class="main-board-wrapper">
      <div class="coordinate-col">
        <div v-for="r in boardSize" :key="'row-coord-' + r" class="coordinate-cell">{{ r - 1 }}</div>
      </div>

      <div
        class="board-grid"
        :style="{ gridTemplateColumns: `repeat(${boardSize}, 1fr)` }"
      >
        <BoardSquare
          v-for="index in boardSize * boardSize"
          :key="index"
          :piece="flatBoard[index - 1]"
          :coords="indexToCoords(index - 1)"
          :is-legal-move="isLegal(indexToCoords(index - 1))"
          :is-last-move="isLast(indexToCoords(index - 1))"
          @make-move="handleMove"
        />
      </div>
    </div>
  </div>
</template>

<script>
import BoardSquare from './BoardSquare.vue';

export default {
  name: 'OthelloBoard',
  components: {
    BoardSquare
  },
  props: {
    board: { type: Array, required: true }, 
    boardSize: { type: Number, required: true },
    legalMoves: { type: Array, default: () => [] }, 
    lastMove: { type: Object, default: () => null } 
  },
  computed: {
    flatBoard() {
      // 展平 2D 数组为 1D 数组，便于迭代
      return this.board.flat();
    }
  },
  methods: {
    indexToCoords(index) {
      // 将一维索引转换为二维坐标 {x, y}
      const x = Math.floor(index / this.boardSize);
      const y = index % this.boardSize;
      return { x, y };
    },
    isLegal({ x, y }) {
      // 检查坐标是否在合法移动列表中
      return this.legalMoves.some(move => move.x === x && move.y === y);
    },
    isLast({ x, y }) {
      // 检查是否是上一步移动
      return this.lastMove && this.lastMove.x === x && this.lastMove.y === y;
    },
    handleMove(coords) {
      this.$emit('move-made', coords);
    }
  }
};
</script>

<style scoped>
.othello-board-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 20px;
  /* 确保棋盘整体宽度不超过设定值 */
  width: 90%; 
  max-width: 630px; 
}

.coordinate-row, .main-board-wrapper {
  display: flex;
}

/* 顶部坐标 */
.coordinate-row {
    width: 92%;
	margin-left: 8%;
	margin-bottom: 3%;
}
.coordinate-row .coordinate-cell {
  flex-grow: 1;
  text-align: center;
  font-weight: bold;
}
.empty-corner {
  width: 30px; /* 与左侧坐标列宽度保持一致 */
  flex-shrink: 0;
}

/* 主板子和侧边坐标 */
.main-board-wrapper {
    width: 100%;
}

.coordinate-col {
  display: flex;
  flex-direction: column;
  width: 11.5%; 
  height: 0;
  margin-top: 5%;
  text-align: center;
  font-weight: bold;
  flex-shrink: 0;
}
.coordinate-col .coordinate-cell {
  /* 使用 padding-bottom 来保持高度与正方形单元格一致 */
  height: 0;
  padding-bottom: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.board-grid {
  display: grid;
  flex-grow: 1;
  width: calc(100% - 30px); /* 100% 减去左侧坐标宽度 */
  border: 2px solid #333;
}
</style>