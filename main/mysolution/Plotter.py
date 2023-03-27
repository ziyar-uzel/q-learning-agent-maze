import math
import matplotlib.pyplot as plt
from main.Maze import Maze
from main.Agent import Agent
from main.QLearning import QLearning
from main.mysolution.MyQLearning import MyQLearning
from main.mysolution.MyEGreedy import MyEGreedy
class Plotter:
    def __init__(self):
        pass
    def optimized_epoch_number(x_test, y_test):

        q = []
        for i in range(1, 151):
            nn = ANN_Draft(0.001, i, 20)
            nn.backpropagate(y_train, x_train)
            a, b, c, d = nn.feed_forward(x_test)
            pred = np.argmax(d, axis=1).astype(np.float32)
            # np.set_printoptions(threshold=np.inf)
            # print(pred+1)
            # print(y_test)
            count_matching = np.count_nonzero((pred + 1) == y_test)
            percentage = count_matching / len(y_test)
            q.append(percentage)

        plt.plot(range(1, 151), q)
        plt.grid()
        plt.title("Validation accuracy curve")
        plt.xlabel("Epochs")
        plt.ylabel("Validation accuracy")
        plt.show()
    def one_run(self,):
        # load the maze
        # TODO replace this with the location to your maze on your file system
        file = "..\\..\\data\\toy_maze.txt"
        maze = Maze(file)
        # Set the reward at the bottom right to 10
        maze.set_reward(maze.get_state(9, 9), 10)

        # create a robot at starting and reset location (0,0) (top left)
        robot = Agent(0, 0)
        # make a selection object (you need to implement the methods in this class)
        selection = MyEGreedy()

        # make a Qlearning object (you need to implement the methods in this class)
        learn = MyQLearning()

        stop = False
        e = 0
        e_decay = 0.99
        step_size = 30000
        # keep learning until you decide to stop
        # while not stop:
        for i in range(10):
            # TODO implement the action selection and learning cycle
            steps_done = False
            for step in range(step_size):
                current_state = robot.get_state(maze)
                chosen_action = selection.get_egreedy_action(agent=robot, maze=maze, q_learning=learn, epsilon=e)
                updated_state = robot.do_action(chosen_action, maze)
                if updated_state is maze.rewards:
                    print("step: " + str(step))
                    break
                # learn.update_q(state=current_state, action=chosen_action, r=maze.get_reward(updated_state), state_next=updated_state, possible_actions=maze.get_valid_actions())
            last_number_of_actions = robot.nr_of_actions_since_reset
            e = e * e_decay
            robot.reset()
        return last_number_of_actions
    def one_run_constant_epsulon(self,alpha_val):
        # load the maze
        # TODO replace this with the location to your maze on your file system
        file = "..\\..\\data\\easy_maze.txt"
        maze = Maze(file)
        # Set the reward at the bottom right to 10
        maze.set_reward(maze.get_state(9, 9), 10)

        # create a robot at starting and reset location (0,0) (top left)
        robot = Agent(0, 0)
        # make a selection object (you need to implement the methods in this class)
        selection = MyEGreedy()

        # make a Qlearning object (you need to implement the methods in this class)
        learn = MyQLearning()

        stop = False
        e = 0.1
        step_size = 30000
        alpha = alpha_val
        gamma = 0.9
        # keep learning until you decide to stop
        # while not stop:
        steps = []
        count = 0
        arr = []
        for i in range(200):
            # TODO implement the action selection and learning cycle
            for step in range(step_size):
                current_state = robot.get_state(maze)
                chosen_action = selection.get_egreedy_action(agent=robot, maze=maze, q_learning=learn, epsilon=e)
                updated_state = robot.do_action(chosen_action, maze)
                # print(updated_state.__str__())
                q = learn.update_q(state=current_state, action=chosen_action, r=maze.get_reward(updated_state),
                                   state_next=updated_state, possible_actions=maze.get_valid_actions(robot), alfa=alpha,
                                   gamma=gamma)
                # print(q)
                if updated_state in maze.rewards:
                    if len(steps) == 10:
                        steps.insert(0, step)
                        steps.pop()
                    else:
                        steps.append(step)
                    break
            if len(steps) == steps.count(steps[0]) and len(steps) == 10:
                break
            last_number_of_actions = robot.nr_of_actions_since_reset
            arr.append(last_number_of_actions)
            robot.reset()
        return arr
    def one_run_epsulon_decay(self,gamma_value):
        # load the maze
        # TODO replace this with the location to your maze on your file system
        file = "..\\..\\data\\toy_maze.txt"
        maze = Maze(file)
        # Set the reward at the bottom right to 10
        maze.set_reward(maze.get_state(9, 9), 10)
        maze.set_reward(maze.get_state(9, 0), 5)

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
        gamma = gamma_value
        # keep learning until you decide to stop
        # while not stop:
        steps = []
        arr = []
        for i in range(500):
            # TODO implement the action selection and learning cycle

            for step in range(step_size):
                current_state = robot.get_state(maze)
                chosen_action = selection.get_egreedy_action(agent=robot, maze=maze, q_learning=learn, epsilon=e)
                updated_state = robot.do_action(chosen_action, maze)
                q = learn.update_q(state=current_state, action=chosen_action, r=maze.get_reward(updated_state),
                                   state_next=updated_state, possible_actions=maze.get_valid_actions(robot), alfa=alpha,
                                   gamma=gamma)
                # print(q)
                if updated_state in maze.rewards:
                    # print(updated_state.__str__())
                    # # print("==============")
                    # if len(steps) == 10:
                    #     steps.insert(0, step)
                    #     steps.pop()
                    # else:
                    #     steps.append(step)
                    break
            # if len(steps) == steps.count(steps[0]) and len(steps) == 10:
            #     break
            last_number_of_actions = robot.nr_of_actions_since_reset
            e = e * e_decay
            arr.append(last_number_of_actions)
            robot.reset()
        return arr
if __name__ == "__main__":
    graph = Plotter()

    # lists_of_lists = []
    # for i in range(0,10):
    #     lists_of_lists.append(graph.one_run_constant_epsulon())
    # q = [sum(x)/10 for x in zip(*lists_of_lists)]

    for gamma_value in {1}:
        q = graph.one_run_epsulon_decay(gamma_value)

        plt.plot(range(1, 501), q)
        plt.grid()
        plt.title("Toy Maze/Gamma Value: " + str(gamma_value))
        plt.xlabel("Number of trial")
        plt.ylabel("Average number of steps")
        plt.show()


    # for alpha_value in {0,0.25,0.50,0.75,1}:
    #
    #     q = graph.one_run_constant_epsulon(alpha_value)
    #
    #     plt.plot(range(1, 201), q)
    #     plt.grid()
    #     plt.title("Easy Maze/Alpha: "+ str(alpha_value))
    #     plt.xlabel("Number of trial")
    #     plt.ylabel("Average number of steps")
    #     plt.show()

