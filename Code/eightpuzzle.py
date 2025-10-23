# eightpuzzle.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import search
import random
import time

# Module Classes

class EightPuzzleState:
    """
    The Eight Puzzle is described in the course textbook on
    page 64.

    This class defines the mechanics of the puzzle itself.  The
    task of recasting this puzzle as a search problem is left to
    the EightPuzzleSearchProblem class.
    """

    def __init__( self, numbers ):
        """
          Constructs a new eight puzzle from an ordering of numbers.

        numbers: a list of integers from 0 to 8 representing an
          instance of the eight puzzle.  0 represents the blank
          space.  Thus, the list

            [1, 0, 2, 3, 4, 5, 6, 7, 8]

          represents the eight puzzle:
            -------------
            | 1 |   | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |
            ------------

        The configuration of the puzzle is stored in a 2-dimensional
        list (a list of lists) 'cells'.
        """
        self.cells = []
        numbers = numbers[:] # Make a copy so as not to cause side-effects.
        numbers.reverse()
        for row in range( 3 ):
            self.cells.append( [] )
            for col in range( 3 ):
                self.cells[row].append( numbers.pop() )
                if self.cells[row][col] == 0:
                    self.blankLocation = row, col

    def isGoal( self ):
        """
          Checks to see if the puzzle is in its goal state.

            -------------
            |   | 1 | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |
            -------------

        >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]).isGoal()
        True

        >>> EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8]).isGoal()
        False
        """
        goal = [[1,2,3],[8,0,4],[7,6,5]]
        for row in range( 3 ):
            for col in range( 3 ):
                if goal[row][col] != self.cells[row][col]:
                    return False
        return True

    def legalMoves( self ):
        """
          Returns a list of legal moves from the current state.

        Moves consist of moving the blank space up, down, left or right.
        These are encoded as 'up', 'down', 'left' and 'right' respectively.

        >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]).legalMoves()
        ['down', 'right']
        """
        moves = []
        row, col = self.blankLocation
        if(row != 0):
            moves.append('up')
        if(row != 2):
            moves.append('down')
        if(col != 0):
            moves.append('left')
        if(col != 2):
            moves.append('right')
        return moves

    def result(self, move):
        """
          Returns a new eightPuzzle with the current state and blankLocation
        updated based on the provided move.

        The move should be a string drawn from a list returned by legalMoves.
        Illegal moves will raise an exception, which may be an array bounds
        exception.

        NOTE: This function *does not* change the current object.  Instead,
        it returns a new object.
        """
        row, col = self.blankLocation
        if(move == 'up'):
            newrow = row - 1
            newcol = col
        elif(move == 'down'):
            newrow = row + 1
            newcol = col
        elif(move == 'left'):
            newrow = row
            newcol = col - 1
        elif(move == 'right'):
            newrow = row
            newcol = col + 1
        else:
            raise "Illegal Move"

        # Create a copy of the current eightPuzzle
        newPuzzle = EightPuzzleState([0, 0, 0, 0, 0, 0, 0, 0, 0])
        newPuzzle.cells = [values[:] for values in self.cells]
        # And update it to reflect the move
        newPuzzle.cells[row][col] = self.cells[newrow][newcol]
        newPuzzle.cells[newrow][newcol] = self.cells[row][col]
        newPuzzle.blankLocation = newrow, newcol

        return newPuzzle

    # Utilities for comparison and display
    def __eq__(self, other):
        """
            Overloads '==' such that two eightPuzzles with the same configuration
          are equal.

          >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]) == \
              EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8]).result('left')
          True
        """
        for row in range( 3 ):
            if self.cells[row] != other.cells[row]:
                return False
        return True

    def __hash__(self):
        return hash(str(self.cells))

    def __getAsciiString(self):
        """
          Returns a display string for the maze
        """
        lines = []
        horizontalLine = ('-' * (13))
        lines.append(horizontalLine)
        for row in self.cells:
            rowLine = '|'
            for col in row:
                if col == 0:
                    col = ' '
                rowLine = rowLine + ' ' + col.__str__() + ' |'
            lines.append(rowLine)
            lines.append(horizontalLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()

# TODO: Implement The methods in this class

class EightPuzzleSearchProblem(search.SearchProblem):
    """
      Implementation of a SearchProblem for the  Eight Puzzle domain

      Each state is represented by an instance of an eightPuzzle.
    """
    def __init__(self,puzzle):
        "Creates a new EightPuzzleSearchProblem which stores search information."
        self.puzzle = puzzle

    def getStartState(self):
        return self.puzzle

    def isGoalState(self,state):
        return state.isGoal()

    def getSuccessors(self,state):
        """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is either left, right, up, or down
          from the original state and the cost is 1.0 for each
        """
        succ = []
        for a in state.legalMoves():
            succ.append((state.result(a), a, 1))
        return succ

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)

EIGHT_PUZZLE_DATA = [[1, 0, 2, 3, 4, 5, 6, 7, 8],
                     [1, 7, 8, 2, 3, 4, 5, 6, 0],
                     [4, 3, 2, 7, 0, 5, 1, 6, 8],
                     [5, 1, 3, 4, 0, 2, 6, 7, 8],
                     [1, 2, 5, 7, 6, 8, 0, 4, 3],
                     [0, 3, 1, 6, 8, 2, 7, 5, 4]]

def loadEightPuzzle(puzzleNumber):
    """
      puzzleNumber: The number of the eight puzzle to load.

      Returns an eight puzzle object generated from one of the
      provided puzzles in EIGHT_PUZZLE_DATA.

      puzzleNumber can range from 0 to 5.

      >>> print(loadEightPuzzle(0))
      -------------
      | 1 |   | 2 |
      -------------
      | 3 | 4 | 5 |
      -------------
      | 6 | 7 | 8 |
      -------------
    """
    return EightPuzzleState(EIGHT_PUZZLE_DATA[puzzleNumber])

def createRandomEightPuzzle(moves=100):
    """
      moves: number of random moves to apply

      Creates a random eight puzzle by applying
      a series of 'moves' random moves to a solved
      puzzle.
    """
    puzzle = EightPuzzleState([1,2,3,8,0,4,7,6,5])
    for i in range(moves):
        # Execute a random legal move
        puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
    return puzzle

import argparse
import ast
import os
from typing import List, Tuple

def parse_board(board_str: str) -> List[List[int]]:
    """
    Accepts strings like: '[[-,2,3],[1,4,5],[8,7,6]]' or '[[-, 2, 3], [1, 4, 5], [8, 7, 6]]'
    Converts '-' to 0 and returns a 3x3 list of ints.
    """
    # Replace '-' with 0, then use ast.literal_eval for safety
    cleaned = board_str.replace('-', '0')
    board = ast.literal_eval(cleaned)
    # Validate 3x3
    if not (isinstance(board, list) and len(board) == 3 and all(isinstance(r, list) and len(r) == 3 for r in board)):
        raise ValueError("Board must be a 3x3 list, e.g. '[[-,2,3],[1,4,5],[8,7,6]]'")
    return board

def flatten(board: List[List[int]]) -> List[int]:
    return [board[r][c] for r in range(3) for c in range(3)]

def board_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    return all(a[r][c] == b[r][c] for r in range(3) for c in range(3))

def str_path_as_pairs(initial_state: 'EightPuzzleState', actions: List[str]) -> str:
    """
    Returns '(s1,a1)->(s2,a2)->...->(sg)' formatting.
    States are printed as 2D lists (as allowed in the spec).
    """
    seq = []
    curr = initial_state
    prev_action = None
    # s1 has no action; we still print (s1, a1) per spec; use 'start' for the first action slot
    seq.append(f"({curr.cells}, start)")
    for a in actions:
        curr = curr.result(a)
        seq.append(f"({curr.cells}, {a})")
    return " -> ".join(seq)

def make_output_filename(search_name: str, heuristic_name: str) -> str:
    # Follow the exact naming convention from the PDF
    # DFS: output DFS.txt; BFS: output BFS.txt; IDS: output IDS.txt; UCS: output UCS.txt
    # GBS, misplaced: output GBS misplaced.txt; GBS, manhattan: output GBS manhattan.txt; GBS, other: output GBS other.txt
    # A*, misplaced: output A* misplaced.txt; etc.
    base = search_name
    if search_name in ("GBS", "ASTAR") and heuristic_name:
        return f"output {base} {heuristic_name}.txt"
    else:
        return f"output {base}.txt"

def make_problem(start_board: List[List[int]], goal_board: List[List[int]]) -> 'EightPuzzleSearchProblem':
    # Patch the problem class so it knows the goal we want to use.
    class EightPuzzleProblemWithGoal(EightPuzzleSearchProblem):
        def __init__(self, puzzle, goal):
            super().__init__(puzzle)
            self.goal = goal  # 2D list

        def isGoalState(self, state):
            return board_equal(state.cells, self.goal)
    return EightPuzzleProblemWithGoal(EightPuzzleState(flatten(start_board)), goal_board)

def make_heuristics(goal_board: List[List[int]]):
    # Rebuild heuristics that use the provided goal instead of a hardcoded one.
    goal_pos = {goal_board[r][c]: (r, c) for r in range(3) for c in range(3)}

    def misplaced(state, problem=None):
        cnt = 0
        for r in range(3):
            for c in range(3):
                v = state.cells[r][c]
                if v != 0 and (r, c) != goal_pos[v]:
                    cnt += 1
        return cnt

    def manhattan(state, problem=None):
        total = 0
        for r in range(3):
            for c in range(3):
                v = state.cells[r][c]
                if v != 0:
                    gr, gc = goal_pos[v]
                    total += abs(r - gr) + abs(c - gc)
        return total

    def other(state, problem=None):
        # Simple weighted combo (you can customize for your “other” heuristic)
        m1 = misplaced(state, problem)
        m2 = manhattan(state, problem)
        return m2 + 0.1 * m1

    return {"misplaced": misplaced, "manhattan": manhattan, "other": other}

def run_and_write(search_name: str,
                  heuristic_name: str,
                  problem: 'EightPuzzleSearchProblem',
                  heur_map,
                  start_state: 'EightPuzzleState'):
    # Map search name to function
    search_name = search_name.upper()
    if search_name == "BFS":
        runner = search.breadthFirstSearch
        heur_fn = None
    elif search_name == "DFS":
        runner = search.depthFirstSearch
        heur_fn = None
    elif search_name == "IDS":
        runner = search.iterativeDeepeningSearch
        heur_fn = None
    elif search_name == "UCS":
        runner = search.uniformCostSearch
        heur_fn = None
    elif search_name in ("GBS", "ASTAR"):
        # require heuristic
        if heuristic_name not in heur_map:
            raise ValueError(f"{search_name} requires a heuristic in {list(heur_map.keys())}")
        heur_fn = heur_map[heuristic_name]
        runner = search.greedyBestFirstSearch if search_name == "GBS" else search.aStarSearch
    else:
        raise ValueError(f"Unknown search: {search_name}")

    # Run + time
    t0 = time.time()
    if heur_fn:
        actions, cost, depth, expanded = runner(problem, heuristic=heur_fn)
    else:
        actions, cost, depth, expanded = runner(problem)
    dt = time.time() - t0

    # Path string
    path_str = str_path_as_pairs(start_state, actions)

    # Write output file per spec
    out_name = make_output_filename(search_name, heuristic_name)
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(f"{search_name}\n")
        f.write("**********\n")
        f.write(f"Path: {path_str}\n")
        f.write(f"Path cost: {cost}\n")
        f.write(f"Depth: {depth}\n")
        f.write(f"Time (s): {dt:.6f}\n")
        f.write(f"Nodes expanded: {expanded}\n")

    return out_name, (cost, depth, dt, expanded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="8-puzzle runner")
    parser.add_argument('--search', required=True,
                        help='Slash-separated list like "BFS/DFS/IDS/UCS/GBS/A*/Beam" (Beam optional)')
    parser.add_argument('--initial', required=True,
                        help="e.g. '[[-,2,3],[1,4,5],[8,7,6]]' (use '-' for blank)")
    parser.add_argument('--goal', required=True,
                        help="e.g. '[[1,2,3],[8,-,4],[7,6,5]]'")
    parser.add_argument('--heuristic', default="",
                        help='For GBS/A*: "misplaced/manhattan/other" (can be slash-separated to run multiple).')
    args = parser.parse_args()

    start_board = parse_board(args.initial)
    goal_board  = parse_board(args.goal)

    # Build problem with your chosen start/goal
    problem = make_problem(start_board, goal_board)
    start_state = problem.puzzle  # convenience

    heur_map = make_heuristics(goal_board)

    # Split selections
    searches = [s.strip() for s in args.search.split('/') if s.strip()]
    heuristics = [h.strip() for h in args.heuristic.split('/') if h.strip()] if args.heuristic else [""]

    # Run everything requested
    summary_rows = []
    made_files = []

    for sname in searches:
        s_upper = sname.upper()
        if s_upper in ("GBS", "ASTAR"):
            to_run = heuristics
        else:
            to_run = [""]  # no heuristic used
        for hname in to_run:
            outfile, metrics = run_and_write(sname, hname, problem, heur_map, start_state)
            made_files.append(outfile)
            summary_rows.append((sname, hname or "-",) + metrics)