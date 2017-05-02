import pika

# ######################### 生产者 #########################
def send(context):
    # 连接到rabbitmq服务器
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='121.42.197.31'))
    channel = connection.channel()

    channel.basic_publish(exchange='', routing_key='respond_execute', body=context)

    print("[Respond] Send:", context)

    # 关闭连接
    connection.close()

# send("Finish")
