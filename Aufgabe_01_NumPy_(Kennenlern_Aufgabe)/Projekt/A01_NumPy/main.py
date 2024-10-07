# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

a_1 = np.arange(100, 201)
# Ausgabe: [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
#  118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
#  136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153
#  154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171
#  172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189
#  190 191 192 193 194 195 196 197 198 199 200]

# zahlen_2 = np.linspace(100, 200, 50, dtype=int)
# Ohne dtype:int wird Float ausgegeben und mit dtype:int wird falsch in Integer umgewandelt
# Ausgabe: [100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134
#  136 138 140 142 144 146 148 151 153 155 157 159 161 163 165 167 169 171
#  173 175 177 179 181 183 185 187 189 191 193 195 197 200]
# Man könnte aber auch arange verwenden
a_2 = np.arange(100, 201, 2)
# Ausgabe: [100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134
#  136 138 140 142 144 146 148 150 152 154 156 158 160 162 164 166 168 170
#  172 174 176 178 180 182 184 186 188 190 192 194 196 198 200]

a_3 = np.arange(100, 110.5, 0.5)
# Ausgabe: [100.  100.5 101.  101.5 102.  102.5 103.  103.5 104.  104.5 105.  105.5
#  106.  106.5 107.  107.5 108.  108.5 109.  109.5 110. ]

a_4_normal = np.random.normal(0, 10, 100).astype(int)
# Ausgabe: [-11   9  -1  -3  10   3 -15  -3 -18  -4  10   4  21   6  -1   7   0   1
#   -4   0  -7 -13   1  -1   0   5 -11  -6  -9   8   6  -2   1  -2 -15  -5
#   22  18 -15   1  -8   5  13   5 -15   6   2   8   5 -15   0   6   2   5
#    0  -5 -14  -7  -8   1   8   0  -2 -12  -5 -11  12   6 -14  -4  -6   4
#  -13  -7  17  -2   1   8   9   5 -17   0   9   3  -9   3  -5  -9 -10   4
#   -2   7  -6   1 -22   0  -9  11  10   0]
a_4_gleich = np.random.uniform(0, 10, 100).astype(int)
# Ausgabe: [7 6 8 2 8 4 3 6 4 4 3 2 5 8 0 6 7 6 9 4 5 0 9 2 4 0 7 1 3 3 6 2 5 2 7 3 2
#  1 4 5 5 5 1 1 2 0 5 5 8 8 7 4 8 0 1 4 3 7 5 9 5 6 9 8 1 3 6 5 1 1 5 0 4 4
#  6 8 1 4 2 5 3 8 6 9 9 7 5 1 6 2 2 0 6 4 7 0 3 0 3 5]


b_1_mean = np.mean(a_4_normal)
# Ausgabe: -0.59
b_1_median = np.median(a_4_normal)
# Ausgabe: 0.0
b_1_min = np.min(a_4_normal)
# Ausgabe: -26
b_1_max = np.max(a_4_normal)
# Ausgabe: 19
b_1_std_deviation = np.std(a_4_normal)
# Ausgabe: 8.646496400276819

b_3_multiplied = a_4_gleich * 100
# Ausgabe: [400 600   0 900 900 800   0 800 100 400 900 200 400   0 800 200 300 200
#  700   0 400 900 500 300 600 400 200 400 800 900 800   0 500 800 800 100
#  700 100 800 200 200 100 700   0 400 900 700 700 900 200   0 400 600   0
#  100 900 800   0 700 100 900 800 100 600 900 500 300 900 800 100 400 100
#    0 900 100   0 500 300 600 900 800 400 700   0 700 900 200   0 100 200
#  100 800 100   0 100 500   0 700 400 500]

b_4 = a_4_gleich[:10]
# Ausgabe: [0 6 8 1 1 8 2 3 3 0]

b_5 = a_4_gleich[a_4_gleich > 0]
# Ausgabe: [5 1 8 9 4 5 7 9 7 8 7 1 8 9 8 4 6 8 9 2 4 2 3 2 1 9 4 3 8 7 6 1 8 1 9 4 5
#  9 8 9 3 2 4 2 6 4 9 1 8 3 1 6 4 1 4 3 9 7 7 9 5 9 5 2 1 9 7 7 8 9 5 6 6 3
#  3 9 3 1 4 4 3 8 6 8 9 2 4 2 7 7 1]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(a_1)
    print(a_2)
    print(a_3)
    print(a_4_normal)
    print(a_4_gleich)
    print(b_1_mean)
    print(b_1_median)
    print(b_1_min)
    print(b_1_max)
    print(b_1_std_deviation)
    print(b_3_multiplied)
    print(b_4)
    print(b_5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
