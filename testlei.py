class Person(object):
    def Work(self):
        print("I am working！")

    # 不加任何参数直接定义，也是类方法

#
#     # 类方法       第一种方法：加装饰器方法
    @classmethod
    def Think(cls, b):  # 类方法Think必须要带至少1个参数,第一个参数默认为类名，后面可以引用。
        cls.Eat(b)  # 在类方法Think中，调用类方法Eat类方法。
         # 在类方法Think中，调用Work类方法。
        print(b, ",I am Thinking!")
#
    # 属于第二种方法，先定义类方法，至少1个参数，第一个默认为类名。
    def Eat(cls, b):
        print(b + ",I am eating")
#
    Eat = classmethod(Eat)  # 第二种方法：通过内建函数classmethod()来创建类方法。

#     # 静态方法，引用时直接用类名.Sleep()即可。
#     @staticmethod
#     def Sleep():
#         print("I am sleeping")
#
#     # 这种方法是：实例对象调用方法
#     def __scolia__(self):
#         print("scola")
#         return "scola"
#
#     # 实例对象可以访问的私有方法，在类方法中可以相互调用和使用。类不能直接访问或者外部访问。
#     def __good(self):
#         print("good")
#         return "good"
#
#
# # Person.Think("li")
# # Person.Eat("jcy")
# # Person.Work('gh')
aa=Person()
aa.Work()

class Person(object):
    def chi(self):
        print("猫喜欢吃鱼")
c=Person()
c.chi()#猫喜欢吃鱼

