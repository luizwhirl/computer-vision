import cv2 
import matplotlib.pyplot as plt 
from ultralytics import YOLO 
 
# YOLOv8 pre-treinado 
model = YOLO('yolov8m.pt') 
 
video_path = 'bridge.mp4' 
output_path = 'output_bridge.mp4' 
 
# a) processamento e detecçao no video 
cap = cv2.VideoCapture(video_path) 
 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 
 
# armanezando contagem de veiculos por frame, e a lista de frame 
contagem_por_frame = [] 
lista_frames = [] 
frame_atual = 0 
 
total_veiculos = set() 
total_por_classe = {2: set(), 3: set(), 5: set()} 
nomes_classes = {2: 'Carro', 3: 'Moto', 5: 'Onibus'} 

MIN_BUS_AREA_RATIO = 0.02
FRAME_AREA = width * height
 
print("iniciciando o processamento \nisso pode levar algum tempo") 
 
while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break 
     
    # realiza a detecção no frame atual 
    # no dataset COCO, as classes para veículos são:  
    # 2 (car)ddd 
    # 3 (motorcycle) 
    # 5 (bus) 
    # 7 (truck) 
    # nisso, o parâmetro 'classes ignora tudo que nao for veículo 
    resultados = model.track(frame, classes=[2, 3, 5], persist=True, verbose=False, conf=0.40, iou=0.45, tracker="botsort.yaml") 

    if resultados[0].boxes is not None and len(resultados[0].boxes) > 0:
        indices_validos = []
        for i, box in enumerate(resultados[0].boxes):
            if int(box.cls.item()) == 5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if ((x2 - x1) * (y2 - y1)) / FRAME_AREA >= MIN_BUS_AREA_RATIO:
                    indices_validos.append(i)
            else:
                indices_validos.append(i)
        resultados[0].boxes = resultados[0].boxes[indices_validos]

    num_veiculos = len(resultados[0].boxes) # conta veiculo por fram 
    contagem_por_frame.append(num_veiculos)  
    lista_frames.append(frame_atual) 
     
    if resultados[0].boxes.id is not None: 
        ids = resultados[0].boxes.id.cpu().numpy().astype(int) 
        classes = resultados[0].boxes.cls.cpu().numpy().astype(int) 
        for obj_id, obj_cls in zip(ids, classes): 
            total_veiculos.add(obj_id) 
            total_por_classe[obj_cls].add(obj_id) 
     
    # bounding box pronta do yolo 
    frame_anotado = resultados[0].plot() 
     
    cv2.putText(frame_anotado, f"Atual: {num_veiculos}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    cv2.putText(frame_anotado, f"Total: {len(total_veiculos)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
     
    y_inicial = height - 150 
    cv2.putText(frame_anotado, "Total por classe:", (20, y_inicial), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
    for i, (cls_id, nome) in enumerate(nomes_classes.items()): 
        qtd = len(total_por_classe[cls_id]) 
        cv2.putText(frame_anotado, f"{nome}: {qtd}", (20, y_inicial + 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
 
    out.write(frame_anotado) 
     
    frame_atual += 1 
 
cap.release() 
out.release() 
print(f"vídeo processado e salvo nessa mesma hora e nesse mesmo canal") 
 
# b) plotagem 
plt.figure(figsize=(12, 6)) 
plt.plot(lista_frames, contagem_por_frame, color='blue', linewidth=1.5, label='Veículos (carros onibus i motas)') 
 
plt.title('quantidade de veículos detectados ao longo do tempo', fontsize=14) 
plt.xlabel('tempo (nº quadro/frame)', fontsize=12) 
plt.ylabel('quantidade de veículos', fontsize=12) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.legend() 
plt.tight_layout() 
 
plt.savefig('grafico.png') 
print("gráfico salvo") 
plt.show()