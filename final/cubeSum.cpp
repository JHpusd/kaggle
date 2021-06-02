# include <iostream>
# include <cassert>
#include <cmath>

int sumFirstNCubes(int n){
    int sum = 0;
    for (int i = 0; i <= n; i++){
        sum += pow(i, 3);
    }
    return sum;
}
int main()
{
    std::cout << sumFirstNCubes(10);
}