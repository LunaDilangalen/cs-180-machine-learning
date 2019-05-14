from sys import argv
from queue import PriorityQueue
import time
from math import sqrt, ceil, floor

def ManhattanDistance(current, goal):
	sum = 0
	row_len = int(sqrt(len(goal)))
	for tile in range(1, len(current)):		# iterate through the tiles in the current state
		# print tile
		i = current.index(tile)		# get the index of the tile in the current state
		j = goal.index(tile)		# get the index of the tile in the goal state

		# row-col coordinates of the tile
		row_i = int(floor(i/row_len))
		col_i = i%row_len
		row_j = int(floor(j/row_len))
		col_j = j%row_len


		sum = sum + (abs(col_i - col_j) + abs(row_i - row_j))
	return sum
			

def LinearConflict(current, goal):
	sum = 0
	row_len = int(sqrt(len(goal)))

	for tile_j in range(1, len(current)):		
		for tile_k in range(tile_j+1, len(current)):		

			#	1. check if tile j and tile k are in the same row or column
			#	2. check if the goal tile j and goal of tile k are in the same row or column
			#	3. check if tile k is in between/or in the goal and/of tile j


			j = current.index(tile_j)		# index of current[j] in the current state
			k = current.index(tile_k)		# index of current[k] in the current state
			j_goal = goal.index(tile_j)		# index of current[j] in the goal state
			k_goal = goal.index(tile_k)		# index of current[k] in the goal state

			# getting their coordinates based on the current state
			row_j = int(floor(j/row_len))
			col_j = j%row_len
			row_k = int(floor(k/row_len))
			col_k = k%row_len

			# getting their coordinates based on the goal state
			row_j_goal = int(floor(j_goal/row_len))
			col_j_goal = j_goal%row_len
			row_k_goal = int(floor(k_goal/row_len))
			col_k_goal = k_goal%row_len

			same_row = (row_j == row_k)
			same_col = (col_j == col_k)
			same_goal_row = (row_j_goal == row_k_goal)
			same_goal_col = (col_j_goal == col_k_goal)
			cond_1 = same_row or same_col 
			cond_2 = same_goal_row or same_goal_col
			
			if(cond_1 and cond_2):
				if(same_row):
					if(col_j_goal <= col_k < col_j):
						# print("1: (", tile_j, tile_k,")")
						sum += 1
				if(same_col):
					if(row_j_goal <= row_k < row_j):
						# print("2: (", tile_j, tile_k,")")
						sum += 1		

	return 2*sum + ManhattanDistance(current, goal)


def TilesOutOfRowCol(current, goal):
	rows, cols = 0, 0
	row_len = int(sqrt(len(goal)))
	# print(row_len)
	for tile in range(1, len(current)):

		i = current.index(tile)
		j = goal.index(tile)

		row_i = int(floor(i/row_len))
		col_i = i%row_len
		row_j = int(floor(j/row_len))
		col_j = j%row_len

		if(row_i != row_j):
			rows+=1
			# print("out of row: %d" %tile)
		if(col_i != col_j):
			cols+=1
			# print("out of col: %d" %tile)

	return rows + cols

def MisplacedTiles(current, goal):
	return sum([1 if current[i] != goal[i] else 0 for i in range(8)])

def Gaschnig(current, goal):
	P = current[:]
	L = []
	# initialize a checker list
	for i in range(1, len(P)):
		L.append(i)

	swaps = 0
	while P != goal:	# know which value x should be placed in the empty tile's position
		i = P.index(0)
		x = goal[i]
		y = P.index(x)
		if(i == goal.index(0)):		# in the case that 0 is in its goal position
			z = P.index(L[0])
			P[z], P[i] = P[i], P[z]	# swap with someone in the checker list
		else:						# get the value with index of x in the current permutation P and swap with P[i]
			P[i], P[y] = P[y], P[i]
			L.remove(x)
			
		swaps += 1
	return swaps

def generateStates(state):
	row_len = row_len = int(sqrt(len(state)))
	possibleMoves = []
	i = state.index(0)

	if row_len == 3:
		if i in [3, 4, 5, 6, 7, 8]:
			new_board = state[:]
			new_board[i], new_board[i - 3] = new_board[i - 3], new_board[i]
			dir = "down"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i in [1, 2, 4, 5, 7, 8]:
			new_board = state[:]
			new_board[i], new_board[i - 1] = new_board[i - 1], new_board[i]
			dir = "right"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i in [0, 1, 3, 4, 6, 7]:
			new_board = state[:]
			new_board[i], new_board[i + 1] = new_board[i + 1], new_board[i]
			dir = "left"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i in [0, 1, 2, 3, 4, 5]:
			new_board = state[:]
			new_board[i], new_board[i + 3] = new_board[i + 3], new_board[i]
			dir = "up"
			temp = [new_board, dir]
			possibleMoves.append(temp)

	elif row_len == 4:
		if i not in [0, 1, 2, 3]:
			new_board = state[:]
			new_board[i], new_board[i - 4] = new_board[i - 4], new_board[i]
			dir = "down"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i not in [0, 4, 8, 12]:
			new_board = state[:]
			new_board[i], new_board[i - 1] = new_board[i - 1], new_board[i]
			dir = "right"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i not in [3, 7, 11, 15]:
			new_board = state[:]
			new_board[i], new_board[i + 1] = new_board[i + 1], new_board[i]
			dir = "left"
			temp = [new_board, dir]
			possibleMoves.append(temp)
		if i not in [12, 13, 14, 15]:
			new_board = state[:]
			new_board[i], new_board[i + 4] = new_board[i + 4], new_board[i]
			dir = "up"
			temp = [new_board, dir]
			possibleMoves.append(temp)

	return possibleMoves

def rebuildPath(closedList, goalState):
	path = []
	end = closedList[str(goalState)]
	while end[5] != "\0":
		path.append(end[5])
		end = closedList[str(end[4])]
	path.reverse()
	return path

def Solver(initialState, goalState, choice):
	openList = PriorityQueue()
	closedList = {}

	# put the initial state in the priority queue
	openList.put([0,0,0,initialState, None, "\0"])		# [f,g,h state, parent]
	generations = 0
	start = time.time()
	while not openList.empty():
		current = openList.get()
		if current[3] == goalState:
			closedList[str(current[3])] = current
			end = time.time()
			path = rebuildPath(closedList, goalState)

			for direction in path:
				print(direction)
			print("time: %f seconds " %(end-start))
			print("generations: %d" %generations)	
			break
		possibleStates = generateStates(current[3])
		for state in possibleStates:
			generations+=1
			# print(state)
			possibleBoard = state[0]
			dir = state[1]
			# h = Heuristic(possibleBoard, goalState, choice)
			if choice == 1:
				h = 0
			elif choice == 2:
				h = ManhattanDistance(possibleBoard, goalState)
			elif choice == 3:
				h = LinearConflict(possibleBoard, goalState)
			elif choice == 4:
				h = MisplacedTiles(possibleBoard, goalState)
			elif choice == 5:
				h = TilesOutOfRowCol(possibleBoard, goalState)
			elif choice == 6:
				h = Gaschnig(possibleBoard, goalState)
			g = current[1]+1	# g = g_parent + 1
			f = g+h
			if str(possibleBoard) not in closedList:
				openList.put([f,g,h, possibleBoard, current[3], dir])
			else:
				# print("f: ", f, " f': ", closedList[str(possibleBoard)][0])
				if(f < closedList[str(possibleBoard)][0]):
					openList.put([f,g,h, possibleBoard, current[3], dir])
		closedList[str(current[3])] = current
	else:
		print("lol no solution!")

def main(argv):
	f = open(argv[1], 'r')
	game = []
	with open(argv[1], 'r') as f:
		game = list(map(int, f.read().split()))

	game_len_half = int(len(game)/2)
	initial = game[:game_len_half]
	goal = game[game_len_half: len(game)]

	print("Start:" , initial)
	print("Goal", goal)
		
	L = [	"No Heuristic",
			"Manhattan Distance",
			"Linear Conflict",
			"Misplaced Tiles",
			"Tiles out of row or column",
			"Gaschnig's"
		]
	for i in range(len(L)):
		print("%d - %s" %(i+1, L[i]))
	choice = int(input("Heuristic to use: "))
	print(L[choice-1])
	Solver(initial, goal, choice)

	
main(argv)

