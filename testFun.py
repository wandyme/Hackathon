from multiprocessing import Manager, Pool
def test(i,lists):
    print(i)
    lists.append(i)
# if __name__=="__main__":
pool=Pool(2)
for i in range(1000):
    lists=Manager().list()
    if len(lists)<=0:  
        pool.apply_async(test,args=(i,lists)) ##需要将lists对象传递给子进程，这里比较耗资源，原因可能是因为Manager类是基于通信的。
    else:
         break
pool.close()
pool.join()
print(lists)