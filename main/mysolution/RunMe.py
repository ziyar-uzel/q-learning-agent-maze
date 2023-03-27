from main.Maze import Maze
from main.Agent import Agent
from main.QLearning import QLearning
from main.mysolution.MyQLearning import MyQLearning
from main.mysolution.MyEGreedy import MyEGreedy

if __name__ == "__main__":
    # load the maze
    # TODO replace this with the location to your maze on your file system
    file = "..\\..\\data\\toy_maze.txt"
    maze = Maze(file)
    # Set the reward at the bottom right to 10
    maze.set_reward(maze.get_state(9, 9), 10)
    # maze.set_reward(maze.get_state(9, 0), 5)

    # create a robot at starting and reset location (0,0) (top left)
    robot = Agent(0, 0)
    # make a selection object (you need to implement the methods in this class)
    selection = MyEGreedy()

    # make a Qlearning object (you need to implement the methods in this class)
    learn = MyQLearning()

    stop = False
    e = 1
    e_decay = 0.99
    step_size = 30000
    alpha = 0.7
    gamma = 0.1
    # keep learning until you decide to stop
    # while not stop:
    steps = []
    for i in range(500):
        # TODO implement the action selection and learning cycle
        for step in range(step_size):
            current_state = robot.get_state(maze)
            chosen_action = selection.get_egreedy_action(agent=robot, maze= maze, q_learning=learn, epsilon= e)
            updated_state = robot.do_action(chosen_action, maze)
            q = learn.update_q(state=current_state, action=chosen_action, r=maze.get_reward(updated_state), state_next=updated_state, possible_actions=maze.get_valid_actions(robot), alfa= alpha, gamma= gamma)
            if updated_state in maze.rewards:
                if len(steps) == 10:
                    steps.insert(0, step)
                    steps.pop()
                else:
                    steps.append(step)
                break
        # Stopping criteria, if the shortest route repeats more than 10 times, terminate
        if len(steps) == steps.count(steps[0]) and len(steps) == 10:
            break
        last_number_of_actions = robot.nr_of_actions_since_reset
        e = e * e_decay
        robot.reset()
    print("Length of the final path: " + str(last_number_of_actions))

