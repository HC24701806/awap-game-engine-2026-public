import random

def gen(n, m, num_c, num_k, num_b, num_s, num_t, num_r, num_u, num_sh, name):
    symbols = (
        ['C'] * num_c + ['K'] * num_k + ['B'] * num_b + 
        ['S'] * num_s + ['T'] * num_t + ['R'] * num_r + 
        ['U'] * num_u + ['$'] * num_sh + ['b'] * 2
    )
    
    succ = False
    for _ in range(20):
        # 1. Randomly populate the inner grid
        # We start with all '.' and then randomly sprinkle '#' and symbols
        inner_area = n * m
        total_symbols = len(symbols)
        
        # Decide how many internal walls (#) we want (e.g., 20% of the grid)
        num_walls = int((inner_area - total_symbols) * 0.2)
        
        # Create a flattened list of the inner grid content
        flat_grid = symbols + (['#'] * num_walls)
        flat_grid += ['.'] * (inner_area - len(flat_grid))
        random.shuffle(flat_grid)
        
        # Reshape into 2D grid with a border
        grid = [['#' for _ in range(n + 2)] for _ in range(m + 2)]
        idx = 0
        for r in range(1, m + 1):
            for c in range(1, n + 1):
                grid[r][c] = flat_grid[idx]
                idx += 1
        
        # 2. Check Walkability
        # Since symbols are unwalkable, we need to ensure all '.' are connected
        # AND every symbol is adjacent to at least one '.'
        if is_valid(grid, m, n, symbols):
            # 3. Success! Write to file
            with open(f"maps/{name}.txt", "w") as f:
                for row in grid:
                    f.write("".join(row) + "\n")
                f.write("\nSWITCH: turn=250 duration=100\n\n")
            
            n = random.randint(10, 101)
            t = random.randint(400, 1001)
            D = random.randint(50, 301)
            gen_orders(n, t, D, name)
            succ = True
            break
    return succ

def is_valid(grid, m, n, symbols_list):
    # Find all floor tiles and symbol tiles
    floors = []
    required_symbols = []
    for r in range(1, m + 1):
        for c in range(1, n + 1):
            if grid[r][c] == '.':
                floors.append((r, c))
            elif grid[r][c] in set(symbols_list):
                required_symbols.append((r, c))
    
    if not floors: return False

    # BFS to check floor connectivity
    start = floors[0]
    visited = {start}
    queue = [start]
    
    while queue:
        curr_r, curr_c = queue.pop(0)
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = curr_r + dr, curr_c + dc
            if grid[nr][nc] == '.' and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    # If not all floors are reachable from each other, fail
    if len(visited) != len(floors):
        return False
    
    # Check if every symbol is adjacent to at least one reachable floor tile
    for sr, sc in required_symbols:
        has_access = False
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            if (sr + dr, sc + dc) in visited:
                has_access = True
                break
        if not has_access:
            return False
            
    return True

def gen_orders(n, t, D, name):
    ingredients = ["EGG", "NOODLES", "ONIONS", "MEAT", "SAUCE"]
    
    # 1. Generate strictly increasing start times
    # We sample n unique numbers from the range [0, t - D)
    # Note: This requires (t - D) > n. 
    upper_bound = t - D
    if n > upper_bound:
        # Fallback: if range is too small, we just space them out by 1
        start_times = list(range(n))
    else:
        start_times = sorted(random.sample(range(upper_bound), n))

    with open(f"maps/{name}.txt", "a") as f:
        f.write("ORDERS:\n")
        
        for i in range(n):
            s = start_times[i]
            
            # d in [0, D]
            d = random.randint(0, D)
            
            # E is a random subset of ingredients
            # Choosing a subset of random size (1 to 5)
            subset_size = random.randint(1, len(ingredients))
            e_subset = random.sample(ingredients, subset_size)
            e_str = ",".join(e_subset)
            
            # r in [0, 100000] (multiples of 1000)
            r = random.randrange(0, 100001, 1000)
            
            # p in [0, 10000] (multiples of 100)
            p = random.randrange(0, 10001, 100)
            
            f.write(f"start={s}\tduration={d}\trequired={e_str}\treward={r}\tpenalty={p}\n")

i = 1
while i <= 100:
    n = random.randint(6, 49)
    m = random.randint(6, 49)
    while True:
        num_c = random.randint(2, max(int(m * n / 16), 3))
        num_k = random.randint(2, max(int(m * n / 32), 3))
        num_b = random.randint(0, 3)
        num_s = random.randint(2, max(int(m * n / 16), 3))
        num_t = random.randint(1, 3)
        num_r = random.randint(0, 3)
        num_u = 1
        num_sh = 1

        if num_c + num_k + num_b + num_s + num_t + num_r + num_u + num_sh < m * n / 4:
            break

    if gen(n, m, num_c, num_k, num_b, num_s, num_t, num_r, num_u, num_sh, f'map_{i}'):
        print(f'Finished map {i}')
        i += 1