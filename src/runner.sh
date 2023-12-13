#! /bin/bash
echo "run GPU tests on 50*50, 75*75, 100*100, 125*125, 150*150, 200*200"

echo "warm up"
./heat 50 50

echo "tests:"
./heat 50 50
./heat 75 75
./heat 100 100
./heat 125 125
./heat 150 150
./heat 200 200