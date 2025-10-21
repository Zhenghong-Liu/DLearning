<template>
  <div class="game-controls">
    <div class="controls-group">
      <label for="board-size">Board Size:</label>
      <select id="board-size" v-model="selectedSize" :disabled="isProcessing">
        <option v-for="size in availableSizes" :key="size" :value="size">{{ size }}x{{ size }}</option>
      </select>
      
      <label for="first-player">First Player:</label>
      <select id="first-player" v-model="selectedFirstPlayer" :disabled="isProcessing">
        <option :value="1">Human (O)</option>
        <option :value="-1">AI (X)</option>
      </select>
      
      <button @click="startNewGameClick" :disabled="isProcessing">New Game</button>
    </div>

    <div class="status-info">
      <div class="turn-status">
        Current Turn: 
        <span :class="['player-indicator', currentPlayer === -1 ? 'black-turn' : 'white-turn']">
          {{ currentPlayerText }}
        </span>
      </div>
      
      <div class="score-display">
          <p class="score white-score">White (O): <strong>{{ pieceCounts.white }}</strong></p>
          <p class="score black-score">Black (X): <strong>{{ pieceCounts.black }}</strong></p>
      </div>
      
      <p class="game-status">Status: <span>{{ gameStatus }}</span></p>
      
      <button v-if="passAction" @click="$emit('pass-turn')" class="pass-button" :disabled="isProcessing">
        Pass Turn (No Legal Moves)
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'GameControls',
  props: {
    currentPlayer: { type: Number, required: true }, // 1 for White (O), -1 for Black (X)
    boardSize: { type: Number, required: true },
    gameStatus: { type: String, default: 'Ongoing' },
    passAction: { type: Boolean, default: false },
    pieceCounts: { 
      type: Object,
      default: () => ({ white: 0, black: 0}) 
    },
    isProcessing: { type: Boolean, default: false }
  },
  data() {
    return {
      selectedSize: this.boardSize,
      availableSizes: [8], 
      selectedFirstPlayer: 1, // 默认人先手 (1)
    };
  },
  computed: {
    currentPlayerText() {
      return this.currentPlayer === 1 ? 'Human (O)' : 'AI (X)';
    }
  },
  watch: {
    boardSize(newSize) {
      this.selectedSize = newSize;
    }
  },
  methods: {
    startNewGameClick() {
      this.$emit('new-game', {
        size: this.selectedSize,
        firstPlayer: this.selectedFirstPlayer
      });
    }
  }
};
</script>

<style scoped>
.game-controls {
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid #ccc;
  border-radius: 5px;
  background-color: #f9f9f9;
  width: 90%;
  max-width: 650px;
}
.controls-group {
  margin-bottom: 15px;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.status-info {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.turn-status {
    margin-bottom: 5px;
}
.player-indicator {
  font-weight: bold;
  padding: 2px 5px;
  border-radius: 3px;
}
.white-turn {
  color: black;
  background-color: white;
  border: 1px solid black;
}
.black-turn {
  color: white;
  background-color: black;
}
.game-status span {
    font-weight: bold;
    color: #007bff;
}
.score-display {
  display: flex;
  gap: 20px;
  margin: 10px 0;
  font-size: 1.1em;
}
.score {
  margin: 0;
  font-weight: normal;
}
.pass-button {
    background-color: orange;
    color: white;
    border: none;
    padding: 8px 15px;
    cursor: pointer;
    border-radius: 4px;
    margin-top: 10px;
}
</style>