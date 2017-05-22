import pika
import re
import Global_Queue as gq


def receive():
    # 连接到rabbitmq服务器
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='121.42.197.31'))
    channel = connection.channel()

    # 声明消息队列，消息将在这个队列中进行传递。如果队列不存在，则创建
    channel.queue_declare(queue='require_execute', durable=True)

    # 定义一个回调函数来处理，这边的回调函数就是将信息打印出来。
    def callback(ch, method, properties, body):
        body = body.decode()
        gq.SHARE_Q.put(re.sub("\D", "", body))
        print("[Require] Get:", body)

    # 告诉rabbitmq使用callback来接收信息
    channel.basic_consume(callback, queue='require_execute', no_ack=True)
    # no_ack=True表示在回调函数中不需要发送确认标识

    print('[*] Waiting for messages. To exit press CTRL+C')

    # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理。按ctrl+c退出。
    channel.start_consuming()
