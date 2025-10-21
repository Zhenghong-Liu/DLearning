import numpy as np
import torch
import json
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# 导入你的游戏和 AI 模块
from game import OthelloGame #
from alphazero import NNetWrapper, MCTS, dotdict #

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
# 启用 CORS，允许前端（通常在不同端口）访问
CORS(app)

# --- 全局状态和 AI 初始化 ---

# 默认参数 (与 alphazero.py 中的 args 保持一致)
args = dotdict({
    'lr': 0.001,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'numIters': 200,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numItersForTrainExamplesHistory': 20,
    'numMCTSSims': 25,  # 训练时的 MCTS 模拟次数
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'board_size': 8 # 默认 8x8
})

# 游戏和 AI 实例
game = None
nnet = None
mcts = None

# 游戏状态
current_board = None
current_player = 1 # 1: Human (White), -1: AI (Black)
last_move_coords = None
board_size = 8

def init_game_and_ai(n):
    """根据板子大小初始化游戏和 AI 模块"""
    global game, nnet, mcts, board_size
    board_size = n
    log.info(f"Initializing game and AI for {n}x{n} board.")
    game = OthelloGame(n) #
    
    # 注意：AlphaZero 模型训练通常针对固定尺寸。
    # 如果你的模型只支持 8x8，这里需要进行处理或重新训练。
    # 这里我们假设模型支持当前尺寸 n。
    
    # 重新配置 MCTS 参数用于 Play 模式
    play_args = dotdict({
        'numMCTSSims': 200,  # 对战时使用更多的模拟次数
        'cpuct': 1.0,
        'cuda': args.cuda # 继承 CUDA 设置
    })
    
    nnet = NNetWrapper(game, args) #
    # 假设你的模型文件已保存到 './checkpoint/best.pth.tar'
    try:
        load_folder = args.load_folder_file[0]
        load_file = args.load_folder_file[1]
        nnet.load_checkpoint(folder=load_folder, filename=load_file)
        log.info(f"Successfully loaded model from {load_folder}{load_file}")
    except ValueError as e:
        log.error(f"Failed to load model: {e}. AI will likely perform poorly.")
        
    mcts = MCTS(game, nnet, play_args) #


def get_api_moves(board, player):
    """将 getValidMoves 结果从向量转换为 {x, y} 列表"""
    if game is None: return []
    
    valids = game.getValidMoves(board, player) #
    moves_list = []
    # 排除最后一个动作（Pass动作）
    for i in range(len(valids) - 1): #
        if valids[i] == 1:
            x = i // game.n
            y = i % game.n
            moves_list.append({'x': int(x), 'y': int(y)})
    return moves_list

def check_game_end(board, player):
    """检查游戏是否结束，并返回状态信息，基于绝对的棋子数量差异。"""
    
    # 获取游戏结束的相对结果 (1: player 赢, -1: player 输, 0: 平局)
    # 注意：这个结果是相对于传入的 player 而言的
    result = game.getGameEnded(board, player) #
    
    status = 'Ongoing'
    score_diff = 0
    
    if result is not None:
        # 获取白棋 (1) 和黑棋 (-1) 的绝对分数。
        # 这里的 score_diff 是：(白棋数量 - 黑棋数量)
        white_count = np.sum(board == 1)
        black_count = np.sum(board == -1)
        score_diff = int(white_count - black_count)
        
        if result == 0:
            status = f"Game Over: Draw. Score: {white_count} vs {black_count}"
        elif score_diff > 0:
            # 白棋 (Human) 数量多，人赢
            status = f"Game Over: Human (O) Wins! Score: {white_count} vs {black_count}"
        elif score_diff < 0:
            # 黑棋 (AI) 数量多，AI 赢
            status = f"Game Over: AI (X) Wins! Score: {white_count} vs {black_count}"
        else:
            # 理论上 result != 0 时分数不会为 0，但以防万一
            status = f"Game Over: Draw. Score: {white_count} vs {black_count}"
    
    return status

@app.route('/api/game/new', methods=['POST'])
def new_game():
    global current_board, current_player, last_move_coords, board_size
    data = request.json
    size = data.get('size', 8)
    
    # 【新增代码】接收 first_player 参数，默认为 1 (Human)
    first_player = data.get('first_player', 1) 
    
    if game is None or size != board_size:
        init_game_and_ai(size)
    
    current_board = game.getInitBoard() #
    current_player = first_player # 【修改】使用接收到的 first_player 设置当前玩家
    last_move_coords = None
    
    status = check_game_end(current_board, current_player)
    
    # 【新增逻辑】如果 AI 先手，立即触发 AI 移动
    if current_player == -1 and status == 'Ongoing':
        return ai_move_logic() # 直接调用 AI 逻辑并返回结果
    # 对current_board进行flip，以符合前端显示习惯
    current_board = np.flip(current_board, 0)

    return jsonify({
        'board': current_board.tolist(),
        'legal_moves': get_api_moves(current_board, current_player),
        'current_player': current_player,
        'last_move': last_move_coords,
        'status': status,
    })


# @app.route('/api/game/human_move', methods=['POST'])
# def human_move():
#     """处理人类玩家移动"""
#     global current_board, current_player, last_move_coords
    
#     if current_player != 1 or check_game_end(current_board, current_player) != 'Ongoing':
#         return jsonify({'error': 'Not your turn or game is over'}), 400

#     data = request.json
#     x = data.get('x')
#     y = data.get('y')

#     if x is None or y is None:
#         # 检查是否是 Pass 动作
#         if data.get('action') == 'pass':
#              action = game.n * game.n # Pass action is the last index
#         else:
#              return jsonify({'error': 'Invalid move coordinates'}), 400
#     else:
#         action = game.n * x + y
    
#     valids = game.getValidMoves(current_board, 1) #
#     if valids[action] == 0:
#         return jsonify({'error': 'Illegal move'}), 400
        
#     current_board, current_player = game.getNextState(current_board, 1, action) #
    
#     if action != game.n * game.n:
#         last_move_coords = {'x': x, 'y': y}
    
#     status = check_game_end(current_board, current_player)
    
#     # 如果游戏未结束且轮到 AI (-1)
#     if status == 'Ongoing' and current_player == -1:
#         # 在这里触发 AI 移动
#         return ai_move_logic()
    
#     return jsonify({
#         'board': current_board.tolist(),
#         'legal_moves': get_api_moves(current_board, current_player),
#         'current_player': current_player,
#         'last_move': last_move_coords,
#         'status': status,
#     })

def ai_move_logic():
    """AI 移动的逻辑封装，在 human_move 中调用"""
    global current_board, current_player, last_move_coords
    
    canonical_board = game.getCanonicalForm(current_board, -1) #
    
    # 获取 AI 的最佳动作 (temp=0)
    ai_action = np.argmax(mcts.getActionProb(canonical_board, temp=0)) #
    
    # 更新游戏状态
    current_board, next_player = game.getNextState(current_board, -1, ai_action) #
    current_player = next_player
    
    # 记录 AI 的移动坐标
    if ai_action != game.n * game.n: # 如果不是 Pass 动作
        ai_x = ai_action // game.n
        ai_y = ai_action % game.n
        last_move_coords = {'x': int(ai_x), 'y': int(ai_y)}
    
    status = check_game_end(current_board, current_player)

    # 对current_board进行flip，以符合前端显示习惯
    current_board = np.flip(current_board, 0)
    
    return jsonify({
        'board': current_board.tolist(),
        'legal_moves': get_api_moves(current_board, current_player),
        'current_player': current_player,
        'last_move': last_move_coords,
        'status': status,
    })

# app.py (在 @app.route('/api/game/human_move', methods=['POST']) 路由下)
# 替换原有的 handleHumanMove/human_move 函数

@app.route('/api/game/human_move', methods=['POST'])
def human_move():
    """处理人类玩家移动，并返回给 AI 的中间状态"""
    global current_board, current_player, last_move_coords
    
    if current_player != 1 or check_game_end(current_board, current_player) != 'Ongoing':
        return jsonify({'error': 'Not your turn or game is over'}), 400

    data = request.json
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        # 检查是否是 Pass 动作
        if data.get('action') == 'pass':
             action = game.n * game.n # Pass action is the last index
        else:
             return jsonify({'error': 'Invalid move coordinates'}), 400
    else:
        action = game.n * x + y

    valids = game.getValidMoves(current_board, 1)
    if valids[action] == 0:
        return jsonify({'error': 'Illegal move'}), 400
    
    # 执行人类移动
    current_board, current_player = game.getNextState(current_board, 1, action)
    
    if action != game.n * game.n:
        last_move_coords = {'x': x, 'y': y}
    
    status = check_game_end(current_board, current_player)

    # 对current_board进行flip，以符合前端显示习惯
    # current_board = np.flip(current_board, 0)
    
    # 注意：这里不再包含 AI 移动逻辑，直接返回
    return jsonify({
        'board': current_board.tolist(),
        'legal_moves': get_api_moves(current_board, current_player),
        'current_player': current_player,
        'last_move': last_move_coords,
        'status': status,
    })


# B. 新增 `ai_move` 路由

@app.route('/api/game/ai_move', methods=['POST'])
def ai_move():
    start_time = time.time()
    """触发 AI 移动，并返回最终状态"""
    global current_board, current_player, last_move_coords
    
    if current_player != -1:
        return jsonify({'error': 'Not AI turn'}), 400
        

    canonical_board = game.getCanonicalForm(current_board, -1)
    ai_action = np.argmax(mcts.getActionProb(canonical_board, temp=0))

    # 执行 AI 移动
    current_board, next_player = game.getNextState(current_board, -1, ai_action)
    current_player = next_player
    
    # 记录 AI 的移动坐标
    if ai_action != game.n * game.n:
        ai_x = ai_action // game.n
        ai_y = ai_action % game.n
        last_move_coords = {'x': int(ai_x), 'y': int(ai_y)}
    else:
        last_move_coords = None # AI Pass

    status = check_game_end(current_board, current_player)

    # 控制 AI 最少思考时间为 0.5 秒
    end_time = time.time()
    used_time = end_time - start_time
    if used_time < 0.5:
        time.sleep(0.5 - used_time)  # 确保至少等待0.5秒

    return jsonify({
        'board': current_board.tolist(),
        'legal_moves': get_api_moves(current_board, current_player),
        'current_player': current_player,
        'last_move': last_move_coords,
        'status': status,
    })

if __name__ == '__main__':
    # 初始化一个默认的 8x8 游戏实例
    init_game_and_ai(8)
    log.info("Starting Flask server on port 5001...")
    app.run(host='0.0.0.0', port=5001)