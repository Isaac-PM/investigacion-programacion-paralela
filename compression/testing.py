cols: int = 64
rows: int = 48

matrix = [[(i, j) for i in range(cols)] for j in range(rows)]

# print(matrix)

interval: int = 2

cols_intervals_indexes = [i for i in range(0, cols, interval)]
rows_intervals_indexes = [i for i in range(0, rows, interval)]

top_left_indexes = [
    (i, j) for i in cols_intervals_indexes for j in rows_intervals_indexes
]

# for s in top_left_indexes:
#     print(s)


def get_adjacent_indexes(top_left, interval):
    x, y = top_left
    return [(i, j) for i in range(x, x + interval) for j in range(y, y + interval)]


print(get_adjacent_indexes((0, 0), interval))
