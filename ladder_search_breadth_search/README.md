# Ladder search game

This is a read me for this project.
My solution is breadth first search, in my algo all those words are created first that are one hop away from the start_word (and are also a valid dictionary word) and added into a que. Once all the words have been added, the algo compares the que elements in FIFO order to the goal_order. If the output is not found, the algo computes ladders (two/three hops away or se) from all the elements of the que and so on until the correct ladder is found. If all the possible search results have been exhausted and the goal_word is not found, it is concluded.  I also used certain optimization techniques in order to reduce the search time
