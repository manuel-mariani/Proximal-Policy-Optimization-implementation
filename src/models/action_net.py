from torch import nn

from utils import test_net


class ActionNet(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()

        # Stacked LSTM
        lstm_hidden_size = input_size
        # lstm_hidden_size = 256
        # lstm_stack_size = 4
        # self.internal_state = None
        # self.memory_block = nn.LSTM(input_size, lstm_hidden_size, lstm_stack_size)

        # Action head
        self.action_block = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        # x, self.internal_state = self.memory_block(x, self.internal_state)
        # self.internal_state = tuple((s.detach() for s in self.internal_state))  # Remove from computation graph
        x = self.action_block(x)
        return x

    def reset(self):
        # self.internal_state = None
        pass


if __name__ == "__main__":
    test_net(ActionNet(128, 15), input_size=(1, 128), output_size=(1, 15))
