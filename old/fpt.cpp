#include <iostream>
#include <vector>
#include <string>
using namespace std;

class DotsAndBoxes {
private:
    int size;
    vector<vector<char>> grid;
    vector<vector<bool>> hLines;  // Horizontal lines
    vector<vector<bool>> vLines;  // Vertical lines
    vector<vector<bool>> boxes;   // Track completed boxes
    int player1Score;
    int player2Score;
    bool player1Turn;

public:
    DotsAndBoxes(int n = 3) : size(n), player1Score(0), player2Score(0), player1Turn(true) {
        // Initialize grid with dots
        grid = vector<vector<char>>(2 * size + 1, vector<char>(2 * size + 1, ' '));
        for (int i = 0; i <= 2 * size; i += 2) {
            for (int j = 0; j <= 2 * size; j += 2) {
                grid[i][j] = '.';
            }
        }
        
        // Initialize lines (initially not drawn)
        hLines = vector<vector<bool>>(size + 1, vector<bool>(size, false));
        vLines = vector<vector<bool>>(size, vector<bool>(size + 1, false));
        boxes = vector<vector<bool>>(size, vector<bool>(size, false));
    }

    void printBoard() {
        // Print column numbers
        cout << "  ";
        for (int j = 0; j <= 2 * size; j++) {
            if (j % 2 == 0) cout << (j / 2) % 10 << " ";
            else cout << "  ";
        }
        cout << "\n  ";
        for (int j = 0; j <= 2 * size; j++) {
            cout << "--";
        }
        cout << "\n";

        for (int i = 0; i <= 2 * size; i++) {
            if (i % 2 == 0) {
                // Row number
                cout << (i / 2) % 10 << "|";
            } else {
                cout << "  ";
            }
            
            for (int j = 0; j <= 2 * size; j++) {
                if (i % 2 == 0) {  // Dot or horizontal line row
                    if (j % 2 == 0) {
                        cout << grid[i][j];
                    } else {
                        if (hLines[i/2][(j-1)/2]) cout << "---";
                        else cout << "   ";
                    }
                } else {  // Vertical line row
                    if (j % 2 == 0) {
                        if (vLines[(i-1)/2][j/2]) cout << "|";
                        else cout << " ";
                    } else {
                        if (boxes[(i-1)/2][(j-1)/2]) {
                            cout << (grid[i-1][j-1] == 'A' ? " A " : " B ");
                        } else {
                            cout << "   ";
                        }
                    }
                }
            }
            cout << "\n";
        }
        cout << "Player 1 (A) score: " << player1Score << "\n";
        cout << "Player 2 (B) score: " << player2Score << "\n";
    }

    bool makeMove(int x1, int y1, int x2, int y2) {
        // Validate move
        if (x1 == x2) {  // Vertical line
            if (abs(y1 - y2) != 1) return false;
            if (y1 > y2) swap(y1, y2);
            if (y1 < 0 || y1 >= size || x1 < 0 || x1 > size) return false;
            if (vLines[y1][x1]) return false;  // Line already drawn
            
            vLines[y1][x1] = true;
        } else if (y1 == y2) {  // Horizontal line
            if (abs(x1 - x2) != 1) return false;
            if (x1 > x2) swap(x1, x2);
            if (x1 < 0 || x1 >= size || y1 < 0 || y1 > size) return false;
            if (hLines[y1][x1]) return false;  // Line already drawn
            
            hLines[y1][x1] = true;
        } else {
            return false;  // Invalid move
        }

        // Check for completed boxes
        bool boxCompleted = false;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (!boxes[i][j] && 
                    (i == 0 || vLines[i-1][j]) && 
                    (j == 0 || hLines[i][j-1]) && 
                    (i == size-1 || vLines[i+1][j]) && 
                    (j == size-1 || hLines[i][j+1])) {
                    // Box completed
                    boxes[i][j] = true;
                    if (player1Turn) {
                        player1Score++;
                        grid[2*i+1][2*j+1] = 'A';
                    } else {
                        player2Score++;
                        grid[2*i+1][2*j+1] = 'B';
                    }
                    boxCompleted = true;
                }
            }
        }

        // Change turn only if no box was completed
        if (!boxCompleted) {
            player1Turn = !player1Turn;
        }

        return true;
    }

    bool isGameOver() {
        return (player1Score + player2Score) == (size * size);
    }

    string getCurrentPlayer() {
        return player1Turn ? "Player 1 (A)" : "Player 2 (B)";
    }

    string getWinner() {
        if (player1Score > player2Score) return "Player 1 (A) wins!";
        else if (player2Score > player1Score) return "Player 2 (B) wins!";
        else return "It's a tie!";
    }
};

int main() {
    int size;
    cout << "Enter grid size (number of boxes per side, e.g., 3 for 3x3): ";
    cin >> size;

    DotsAndBoxes game(size);
    
    while (!game.isGameOver()) {
        game.printBoard();
        
        cout << game.getCurrentPlayer() << "'s turn\n";
        cout << "Enter coordinates (x1 y1 x2 y2) to draw a line between two adjacent dots: ";
        
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        
        if (!game.makeMove(x1, y1, x2, y2)) {
            cout << "Invalid move! Try again.\n";
        }
    }
    
    game.printBoard();
    cout << "Game Over! " << game.getWinner() << "\n";
    
    return 0;
}