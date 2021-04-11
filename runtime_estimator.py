
cache_refresh_rate = 8
first_epoch = 6.5
cache_t = 0.3333
test_t = 2.5

epoch_t = first_epoch - cache_t * cache_refresh_rate
epochs = 30
total = 0
for e in range(1, epochs+1):
    if e % 5 == 0 and e > 2:
        cache_refresh_rate /= 2

    t = epoch_t + cache_refresh_rate * cache_t + test_t
    print(f"epoch: {e}, time: {t} mins")
    total += t

print(f"total: {int(total // 60) :0>2}:{int(total % 60) :0>2}")