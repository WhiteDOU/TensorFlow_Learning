from collections import deque
w = ['cat','dog','donkey']

for i in w:
    print(i,i.__len__())

for i in range(w.__len__()):
    print(i,w[i])


def fib(n):
    a,b=0,1
    while a<n:
       print(a)
       a,b=b,a+b

def fix(n):
    n=1
    print(n)

n=2
print(n)
fix(n)
print(n)



queue = deque([1,2,3,4])
print(queue)
queue.append(6)
print((queue))
queue.popleft()
print(queue)

test = list(map(lambda x:x**2,range(10)))
print(test)

x = 123,23,'hello'
print(x)
a,b,c=x
print(a)

#set
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)