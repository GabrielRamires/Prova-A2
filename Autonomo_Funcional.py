import cv2
import numpy as np
import os

# ============================================================
# CONFIGURAÇÃO
# ============================================================
MAIN_SQUARE_SIZE_CM = 80.0
ARUCO_MARKER_SIZE_CM = 16.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_ID = 0

SHAPE_DIAM_MIN_CM = 4.0
SHAPE_DIAM_MAX_CM = 12.0

BLUE_H_MIN = 90
BLUE_H_MAX = 130
BLUE_S_MIN = 60
BLUE_V_MIN = 60

# ============================================================
# REGIÕES COM TEXTO (apagar da máscara)
# Formato: (x, y, largura, altura) em PIXELS
# Deixe vazio [] se estiver testando com rotação
# ============================================================
TEXT_REGIONS = [
    (20, 185, 55, 235),
    (210, 485, 200, 55),
]

BORDA_MARGIN_PX = 30


class HybridShapeDetector:
    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise ValueError(f"Arquivo não encontrado: {image_path}")
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        print(f"✓ Imagem carregada: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        self.original = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        self.main_square = None
        self.aruco_corners = None
        self.aruco_center = None
        
        self.origin_point = None
        self.scale_factor = None
        self.scale_aruco = None
        self.scale_square = None
        self.shapes = []
        
        self.main_size_cm = MAIN_SQUARE_SIZE_CM
        self.aruco_size_cm = ARUCO_MARKER_SIZE_CM
        
        # Referencial de orientação (descoberto a partir do ArUco)
        self.idx_canto_ref = None
        self.eixo_x_vec = None
        self.eixo_y_vec = None

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, blurred

    def detect_all_quadrilaterals(self, image):
        all_quads = []
        
        _, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        all_quads.extend(self.find_quads_in_thresh(thresh1))
        
        thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        all_quads.extend(self.find_quads_in_thresh(thresh2))
        
        _, thresh3 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
        all_quads.extend(self.find_quads_in_thresh(thresh3))
        
        unique_quads = []
        for quad in all_quads:
            is_dup = False
            center_new = np.mean(quad[0], axis=0)
            for existing in unique_quads:
                center_exist = np.mean(existing[0], axis=0)
                if np.linalg.norm(center_new - center_exist) < 20:
                    is_dup = True
                    break
            if not is_dup:
                unique_quads.append(quad)
        
        return unique_quads

    def find_quads_in_thresh(self, thresh):
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            for factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                epsilon = factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = self.order_points(approx.reshape(4, 2))
                    quads.append((pts, area))
                    break
        
        return quads

    def order_points(self, pts):
        """Ordena pontos: SE, SD, ID, IE (relativo à IMAGEM)"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]      # Superior Esquerdo
        rect[2] = pts[np.argmax(s)]      # Inferior Direito
        rect[1] = pts[np.argmin(diff)]   # Superior Direito
        rect[3] = pts[np.argmax(diff)]   # Inferior Esquerdo
        return rect

    def identify_main_square(self, all_quads):
        if len(all_quads) < 1:
            return False
        
        all_quads.sort(key=lambda x: x[1], reverse=True)
        self.main_square = all_quads[0][0]
        main_area = all_quads[0][1]
        
        print(f"✓ Quadrado principal ({self.main_size_cm:.0f}x{self.main_size_cm:.0f}cm) detectado")
        print(f"   Área = {main_area:.0f} pixels²")
        return True

    def detect_main_square(self):
        print("\n🔍 ETAPA 1: Detecção automática do quadrado principal...")
        print("=" * 60)
        
        gray, blurred = self.preprocess_image()
        all_quads = self.detect_all_quadrilaterals(blurred)
        print(f"   Quadriláteros encontrados: {len(all_quads)}")
        
        if not self.identify_main_square(all_quads):
            print("❌ Não foi possível detectar o quadrado principal automaticamente!")
            return False
        
        return True

    def detect_aruco_and_calibrate(self):
        print("\n🎯 ETAPA 2: Detectando ArUco e calibrando escala...")
        print("=" * 60)
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )
        
        if ids is None or len(ids) == 0:
            print("❌ Nenhum marcador ArUco detectado!")
            return False
        
        print(f"   Marcadores ArUco detectados: {len(ids)}")
        for i, marker_id in enumerate(ids.flatten()):
            print(f"      - ID {marker_id} em {corners[i][0].mean(axis=0).astype(int)}")
        
        target_idx = None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ARUCO_ID:
                target_idx = i
                break
        
        if target_idx is None:
            print(f"❌ Marcador ArUco ID {ARUCO_ID} NÃO encontrado!")
            return False
        
        # ⭐ NÃO reordenar! Manter ordem original do OpenCV
        # Isso preserva a informação de orientação do marcador físico
        self.aruco_corners = corners[target_idx][0].astype(np.float32)
        self.aruco_center = self.aruco_corners.mean(axis=0)
        
        print(f"   Cantos do ArUco (ordem do OpenCV):")
        for i, c in enumerate(self.aruco_corners):
            print(f"      canto[{i}]: ({c[0]:.1f}, {c[1]:.1f})")
        
        print("\n📐 CALIBRAÇÃO COMBINADA (ArUco + Quadrado)")
        print("-" * 60)
        
        aruco_sides = [
            np.linalg.norm(self.aruco_corners[(i + 1) % 4] - self.aruco_corners[i])
            for i in range(4)
        ]
        aruco_side_px = float(np.mean(aruco_sides))
        self.scale_aruco = self.aruco_size_cm / aruco_side_px
        
        print(f"ArUco:")
        print(f"   Lado médio: {aruco_side_px:.2f} pixels")
        print(f"   Escala: {self.scale_aruco:.5f} cm/pixel")
        
        sq_sides = [
            np.linalg.norm(self.main_square[(i + 1) % 4] - self.main_square[i])
            for i in range(4)
        ]
        sq_side_px = float(np.mean(sq_sides))
        self.scale_square = self.main_size_cm / sq_side_px
        
        print(f"\nQuadrado 80x80:")
        print(f"   Lado médio: {sq_side_px:.2f} pixels")
        print(f"   Escala: {self.scale_square:.5f} cm/pixel")
        
        diff_pct = abs(self.scale_aruco - self.scale_square) / self.scale_square * 100
        print(f"\nComparação: diferença = {diff_pct:.2f}%")
        
        if diff_pct < 5.0:
            self.scale_factor = (self.scale_aruco + 4.0 * self.scale_square) / 5.0
            print(f"   ✓ Escalas consistentes → média ponderada: {self.scale_factor:.5f} cm/pixel")
        else:
            self.scale_factor = self.scale_square
            print(f"   ⚠️  Divergência → usando quadrado: {self.scale_factor:.5f} cm/pixel")
        
        # ============================================================
        # ORIGEM = CANTO DO QUADRADO MAIS PRÓXIMO DO ARUCO
        # ============================================================
        print(f"\n📍 ORIGEM (0,0):")
        
        distancias = []
        for i, corner in enumerate(self.main_square):
            d = np.linalg.norm(self.aruco_center - corner)
            distancias.append(d)
            print(f"   Canto {i} ({corner[0]:.0f}, {corner[1]:.0f}): dist = {d:.1f} px")
        
        idx_ref = int(np.argmin(distancias))
        canto_ref_square = self.main_square[idx_ref]
        self.idx_canto_ref = idx_ref
        
        nomes_cantos = ["Superior Esquerdo", "Superior Direito",
                        "Inferior Direito", "Inferior Esquerdo"]
        print(f"\n   ⭐ Canto de referência: {nomes_cantos[idx_ref]} (índice {idx_ref})")
        
        # Canto do ArUco correspondente (mais próximo do canto do quadrado)
        distancias_aruco = [np.linalg.norm(canto_ref_square - c) for c in self.aruco_corners]
        idx_canto_aruco = int(np.argmin(distancias_aruco))
        canto_ref_aruco = self.aruco_corners[idx_canto_aruco]
        
        print(f"   ⭐ Canto correspondente no ArUco: índice {idx_canto_aruco} "
              f"({canto_ref_aruco[0]:.0f}, {canto_ref_aruco[1]:.0f})")
        
        dist_entre_cantos = np.linalg.norm(canto_ref_square - canto_ref_aruco)
        print(f"   ⭐ Distância entre os dois cantos: {dist_entre_cantos:.1f} px")
        
        if dist_entre_cantos < 15.0:
            self.origin_point = (canto_ref_square + canto_ref_aruco) / 2.0
            print(f"   ✓ Coincidem → média dos dois")
        else:
            self.origin_point = canto_ref_square
            print(f"   ⚠️  Diferentes → usando canto do QUADRADO")
        
        print(f"   → Origem final: ({self.origin_point[0]:.1f}, {self.origin_point[1]:.1f}) px")
        
        # ============================================================
        # ⭐ ORIENTAÇÃO DOS EIXOS USANDO ROTAÇÃO DO ARUCO
        # ============================================================
        # O OpenCV retorna os cantos do ArUco em ordem consistente
        # relativa ao marcador FÍSICO (independe da rotação da imagem):
        #   canto[0] = superior esquerdo do marcador impresso
        #   canto[1] = superior direito do marcador impresso
        #   canto[2] = inferior direito do marcador impresso
        #   canto[3] = inferior esquerdo do marcador impresso
        #
        # No referencial do PAPEL:
        #   X+ = direção canto[0] → canto[1]
        #   Y+ = direção canto[3] → canto[0] (para cima do marcador)
        
        eixo_x_papel = self.aruco_corners[1] - self.aruco_corners[0]
        eixo_x_papel = eixo_x_papel / np.linalg.norm(eixo_x_papel)
        
        eixo_y_papel = self.aruco_corners[0] - self.aruco_corners[3]
        eixo_y_papel = eixo_y_papel / np.linalg.norm(eixo_y_papel)
        
        # ⚠️ Precisamos garantir que X e Y apontem para DENTRO do quadrado 80x80,
        # pois o ArUco está colado no canto e a origem é o canto desse quadrado.
        # O "X do papel" pode apontar para dentro ou para fora dependendo
        # de qual canto o ArUco está colado.
        
        para_dentro = self.main_square.mean(axis=0) - canto_ref_square
        para_dentro_norm = para_dentro / (np.linalg.norm(para_dentro) + 1e-9)
        
        # Se X apontar para fora, inverte
        if np.dot(eixo_x_papel, para_dentro_norm) < 0:
            eixo_x_papel = -eixo_x_papel
            print(f"   ⚠️  Eixo X estava apontando para fora do quadrado → invertido")
        
        # Se Y apontar para fora, inverte
        if np.dot(eixo_y_papel, para_dentro_norm) < 0:
            eixo_y_papel = -eixo_y_papel
            print(f"   ⚠️  Eixo Y estava apontando para fora do quadrado → invertido")
        
        # ⚠️ Ainda pode ocorrer de X e Y ficarem trocados (X apontando na aresta
        # que deveria ser Y). Isso acontece quando o ArUco está colado em um canto
        # onde a orientação natural do marcador não coincide com o X/Y esperado.
        #
        # Regra adicional: se o canto de referência é um canto "superior" (idx 0 ou 1),
        # o Y+ deve apontar "para baixo na imagem" no referencial do papel; se é "inferior"
        # (idx 2 ou 3), o Y+ deve apontar "para cima na imagem".
        # 
        # Mas como estamos usando o referencial do papel (não da imagem), não importa
        # a posição na imagem — o importante é que X e Y sigam o papel.
        #
        # Deixamos como está: o resultado é o referencial intrínseco do papel.
        
        self.eixo_x_vec = eixo_x_papel
        self.eixo_y_vec = eixo_y_papel
        
        print(f"\n🧭 ORIENTAÇÃO DOS EIXOS (referencial do PAPEL):")
        print(f"   Eixo X+ (unit): ({eixo_x_papel[0]:.3f}, {eixo_x_papel[1]:.3f})")
        print(f"   Eixo Y+ (unit): ({eixo_y_papel[0]:.3f}, {eixo_y_papel[1]:.3f})")
        
        # Debug: mostrar eixos desenhados
        debug = self.original.copy()
        cv2.polylines(debug, [self.main_square.astype(np.int32)], True, (0, 200, 0), 2)
        cv2.polylines(debug, [self.aruco_corners.astype(np.int32)], True, (255, 0, 255), 2)
        
        # Rótulos dos cantos do ArUco
        for i, c in enumerate(self.aruco_corners.astype(int)):
            cv2.circle(debug, tuple(c), 6, (255, 255, 0), -1)
            cv2.putText(debug, f"a{i}", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        origin_int = tuple(self.origin_point.astype(int))
        cv2.circle(debug, origin_int, 10, (0, 0, 255), -1)
        cv2.putText(debug, "ORIGEM", (origin_int[0] + 15, origin_int[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Desenhar eixos
        axis_len = 100
        x_end = (int(origin_int[0] + eixo_x_papel[0] * axis_len),
                 int(origin_int[1] + eixo_x_papel[1] * axis_len))
        y_end = (int(origin_int[0] + eixo_y_papel[0] * axis_len),
                 int(origin_int[1] + eixo_y_papel[1] * axis_len))
        
        cv2.arrowedLine(debug, origin_int, x_end, (255, 0, 0), 3, tipLength=0.15)
        cv2.arrowedLine(debug, origin_int, y_end, (0, 255, 0), 3, tipLength=0.15)
        cv2.putText(debug, "X+", (x_end[0] + 10, x_end[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug, "Y+", (y_end[0] + 10, y_end[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Orientacao dos eixos - pressione tecla", debug)
        print("   >>> Veja a janela 'Orientacao dos eixos' e pressione tecla")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True

    def expand_polygon(self, polygon, margin_px):
        center = polygon.mean(axis=0)
        expanded = []
        for pt in polygon:
            direction = pt - center
            dist = np.linalg.norm(direction)
            if dist > 0:
                new_pt = pt + direction * (margin_px / dist)
            else:
                new_pt = pt
            expanded.append(new_pt)
        return np.array(expanded, dtype=np.float32)

    def pixels_to_cm(self, px, py):
        """
        Converte pixel para cm usando os eixos descobertos.
        Isso garante que X e Y sigam o referencial do PAPEL,
        independente da rotação da imagem.
        """
        ponto_vec = np.array([px - self.origin_point[0],
                              py - self.origin_point[1]])
        
        x_cm = np.dot(ponto_vec, self.eixo_x_vec) * self.scale_factor
        y_cm = np.dot(ponto_vec, self.eixo_y_vec) * self.scale_factor
        
        return (x_cm, y_cm)

    def classify_shape(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        vertices_votes = []
        for factor in [0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(contour, factor * perimeter, True)
            vertices_votes.append(len(approx))
        
        vertices = max(set(vertices_votes), key=vertices_votes.count)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if circularity > 0.82:
            return "Circulo"
        
        if vertices == 3:
            return "Triangulo"
        elif vertices == 4:
            if 0.80 <= aspect_ratio <= 1.25:
                return "Quadrado"
            else:
                return "Retangulo"
        elif vertices == 5:
            return "Pentagono"
        elif vertices == 6:
            return "Hexagono"
        elif vertices >= 7:
            if circularity > 0.75:
                return "Circulo"
            return None
        
        return None

    def detect_shapes_automatically(self):
        print("\n" + "=" * 60)
        print("🤖 ETAPA 3: DETECÇÃO AUTOMÁTICA DAS FORMAS")
        print("=" * 60)
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        margin_px = int(0.05 * np.linalg.norm(self.main_square[1] - self.main_square[0]))
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        center = self.main_square.mean(axis=0)
        inner_square = []
        for corner in self.main_square:
            direction = center - corner
            dist = np.linalg.norm(direction)
            new_corner = corner + direction * (margin_px / dist)
            inner_square.append(new_corner)
        inner_square = np.array(inner_square, dtype=np.float32)
        cv2.fillPoly(mask, [inner_square.astype(np.int32)], 255)
        
        if self.aruco_corners is not None:
            aruco_mask = np.zeros(gray.shape, dtype=np.uint8)
            aruco_expanded = self.expand_polygon(self.aruco_corners, margin_px * 1.2)
            cv2.fillPoly(aruco_mask, [aruco_expanded.astype(np.int32)], 255)
            mask = cv2.subtract(mask, aruco_mask)
        
        # Remover textos (só funciona se a imagem NÃO estiver rotacionada)
        print(f"\n   Removendo {len(TEXT_REGIONS)} regiões de texto:")
        for (tx, ty, tw, th) in TEXT_REGIONS:
            mask[ty:ty+th, tx:tx+tw] = 0
            print(f"      - ({tx}, {ty}, {tw}x{th})")
        
        # Binarização HSV azul
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN])
        upper_blue = np.array([BLUE_H_MAX, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        thresh = cv2.bitwise_and(mask_blue, mask_blue, mask=mask)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        print(f"\n   Binarização via HSV — AZUL")
        
        cv2.imshow("Mask azul (debug)", mask_blue)
        cv2.imshow("Threshold final (debug)", thresh)
        print("   >>> Veja as janelas de debug e pressione tecla")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Faixa de área
        diam_min_px = SHAPE_DIAM_MIN_CM / self.scale_factor
        diam_max_px = SHAPE_DIAM_MAX_CM / self.scale_factor
        area_min_px = np.pi * (diam_min_px / 2) ** 2 * 0.5
        area_max_px = np.pi * (diam_max_px / 2) ** 2 * 3.0
        
        print(f"   Faixa de área: {area_min_px:.0f} a {area_max_px:.0f} px²")
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Contornos encontrados: {len(contours)}")
        
        candidates = []
        rejected = {'area_min': 0, 'area_max': 0, 'posicao': 0, 'borda': 0,
                    'aspect': 0, 'solidity': 0}
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < area_min_px:
                rejected['area_min'] += 1
                continue
            
            if area > area_max_px:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity_check = area / hull_area if hull_area > 0 else 0
                circularity_check = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if not (solidity_check > 0.90 and circularity_check > 0.70):
                    rejected['area_max'] += 1
                    continue
                else:
                    print(f"      ⚠️  Área {area:.0f} > máximo, mas forma regular → aceita")
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            if cv2.pointPolygonTest(inner_square, (cx, cy), False) < 0:
                rejected['posicao'] += 1
                continue
            
            if (cx < BORDA_MARGIN_PX or cx > self.width - BORDA_MARGIN_PX or
                cy < BORDA_MARGIN_PX or cy > self.height - BORDA_MARGIN_PX):
                rejected['borda'] += 1
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect > 2.0:
                rejected['aspect'] += 1
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.80:
                rejected['solidity'] += 1
                continue
            
            candidates.append({
                'contour': contour,
                'area': area,
                'center': (cx, cy),
                'solidity': solidity,
                'aspect': aspect,
            })
        
        print(f"\n   Rejeições por filtro:")
        for r, c in rejected.items():
            if c > 0:
                print(f"      - {r}: {c}")
        
        print(f"   Candidatos válidos: {len(candidates)}")
        
        shapes = []
        for cand in candidates:
            contour = cand['contour']
            area = cand['area']
            cx, cy = cand['center']
            
            shape_type = self.classify_shape(contour)
            if shape_type is None:
                print(f"   ⚠️  Contorno em ({cx},{cy}) área={area:.0f} não classificado")
                continue
            
            x_cm, y_cm = self.pixels_to_cm(cx, cy)
            diam_px = 2 * np.sqrt(area / np.pi)
            diam_cm = diam_px * self.scale_factor
            
            shapes.append({
                'type': shape_type,
                'center_pixels': (cx, cy),
                'center_cm': (round(x_cm, 2), round(y_cm, 2)),
                'area_px': area,
                'diam_cm': round(diam_cm, 2),
                'contour': contour,
                'solidity': round(cand['solidity'], 3),
                'aspect': round(cand['aspect'], 3),
            })
        
        print(f"   Formas classificadas: {len(shapes)}")
        
        if len(shapes) > 4:
            print(f"   ⚠️  Mais de 4 formas! Mantendo as 4 mais confiáveis.")
            def score(s):
                sx, sy = s['center_pixels']
                dist_borda = min(sx, sy, self.width - sx, self.height - sy)
                return -(s['solidity'] * 1000 + dist_borda * 2 + s['area_px'] * 0.05)
            shapes.sort(key=score)
            shapes = shapes[:4]
        elif len(shapes) < 4:
            print(f"   ⚠️  Apenas {len(shapes)} formas detectadas (esperado: 4)")
        
        shapes.sort(key=lambda s: (-s['center_cm'][1], s['center_cm'][0]))
        
        # Debug visual
        debug = self.original.copy()
        cv2.polylines(debug, [self.main_square.astype(np.int32)], True, (0, 200, 0), 2)
        cv2.polylines(debug, [self.aruco_corners.astype(np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(debug, [inner_square.astype(np.int32)], True, (100, 100, 100), 1)
        
        for (tx, ty, tw, th) in TEXT_REGIONS:
            cv2.rectangle(debug, (tx, ty), (tx+tw, ty+th), (0, 0, 255), 2)
        
        for cand in candidates:
            if not any(np.array_equal(cand['contour'], s['contour']) for s in shapes):
                cv2.drawContours(debug, [cand['contour']], -1, (0, 0, 255), 1)
        
        for shape in shapes:
            cv2.drawContours(debug, [shape['contour']], -1, (0, 255, 0), 3)
            cx, cy = shape['center_pixels']
            cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(debug, shape['type'], (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Deteccao automatica - pressione qualquer tecla", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return shapes

    def click_shape_centers(self):
        points = []
        click_img = self.original.copy()
        
        cv2.polylines(click_img, [self.main_square.astype(np.int32)], True, (0, 200, 0), 2)
        if self.aruco_corners is not None:
            cv2.polylines(click_img, [self.aruco_corners.astype(np.int32)], True, (255, 0, 255), 2)
        
        if self.origin_point is not None:
            origin = tuple(self.origin_point.astype(int))
            cv2.circle(click_img, origin, 8, (0, 0, 255), -1)
            cv2.circle(click_img, origin, 11, (255, 255, 255), 2)
            cv2.putText(click_img, "ORIGEM (0,0)",
                        (origin[0] + 15, origin[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        current_type = "Triangulo"
        shape_types = ["Triangulo", "Quadrado", "Retangulo", "Circulo",
                       "Pentagono", "Hexagono", "Outro"]
        type_index = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_type, type_index
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append({'pixel': (x, y), 'type': current_type})
                
                cv2.circle(click_img, (x, y), 6, (0, 255, 255), -1)
                cv2.circle(click_img, (x, y), 9, (255, 255, 255), 2)
                
                if self.origin_point is not None:
                    origin = tuple(self.origin_point.astype(int))
                    cv2.line(click_img, origin, (x, y), (255, 255, 0), 1, cv2.LINE_AA)
                
                cv2.putText(click_img, f"{len(points)}. {current_type}", (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                print(f"   Ponto {len(points)}: {current_type} em ({x}, {y}) pixels")
                
                temp = click_img.copy()
                cv2.putText(temp, f"Forma: {current_type} | ENTER para finalizar | T para mudar tipo",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Clique nos centros das formas", temp)
        
        temp = click_img.copy()
        cv2.putText(temp, f"Forma atual: {current_type} | Clique nos centros | T=mudar tipo | ENTER=finalizar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Clique nos centros das formas", temp)
        cv2.setMouseCallback("Clique nos centros das formas", mouse_callback)
        
        print(f"\n   Tipo atual: {current_type}")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                cv2.destroyAllWindows()
                return None
            elif key == 13:
                cv2.destroyAllWindows()
                break
            elif key == ord('t') or key == ord('T'):
                type_index = (type_index + 1) % len(shape_types)
                current_type = shape_types[type_index]
                print(f"\n   ► Tipo alterado para: {current_type}")
                
                temp = click_img.copy()
                cv2.putText(temp, f"Forma atual: {current_type} | T=mudar | ENTER=finalizar",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Clique nos centros das formas", temp)
        
        for shape in points:
            px, py = shape['pixel']
            x_cm, y_cm = self.pixels_to_cm(px, py)
            shape['center_cm'] = (round(x_cm, 2), round(y_cm, 2))
            shape['diam_cm'] = 0
            shape['area_px'] = 0
            shape['contour'] = None
        
        return points

    def draw_final_result(self):
        result = self.original.copy()
        
        cv2.polylines(result, [self.main_square.astype(np.int32)], True, (0, 200, 0), 2)
        cv2.putText(result, f"Quadrado {self.main_size_cm:.0f}x{self.main_size_cm:.0f}cm",
                    (int(self.main_square[0][0]), int(self.main_square[0][1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        
        if self.aruco_corners is not None:
            cv2.polylines(result, [self.aruco_corners.astype(np.int32)], True, (255, 0, 255), 3)
            cv2.putText(result, f"ArUco ID {ARUCO_ID} ({self.aruco_size_cm:.0f}cm)",
                        (int(self.aruco_corners[0][0]),
                         int(self.aruco_corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        origin = tuple(self.origin_point.astype(int))
        cv2.circle(result, origin, 12, (0, 0, 255), -1)
        cv2.circle(result, origin, 15, (255, 255, 255), 3)
        cv2.putText(result, "ORIGEM (0,0)", (origin[0] + 25, origin[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # Eixos usando os vetores descobertos (independe de rotação)
        axis_len = 150
        x_end = (int(origin[0] + self.eixo_x_vec[0] * axis_len),
                 int(origin[1] + self.eixo_x_vec[1] * axis_len))
        y_end = (int(origin[0] + self.eixo_y_vec[0] * axis_len),
                 int(origin[1] + self.eixo_y_vec[1] * axis_len))
        
        cv2.arrowedLine(result, origin, x_end, (255, 0, 0), 3, tipLength=0.15)
        cv2.arrowedLine(result, origin, y_end, (0, 255, 0), 3, tipLength=0.15)
        
        cv2.putText(result, "X+", (x_end[0] + 10, x_end[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(result, "Y+", (y_end[0] + 10, y_end[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        colors = {
            'Triangulo': (0, 255, 255),
            'Quadrado': (255, 255, 0),
            'Retangulo': (255, 0, 255),
            'Circulo': (0, 255, 0),
            'Pentagono': (255, 128, 0),
            'Hexagono': (128, 0, 255),
            'Outro': (255, 255, 255)
        }
        
        for i, shape in enumerate(self.shapes):
            color = colors.get(shape['type'], (255, 255, 255))
            center = shape['center_pixels']
            
            if shape.get('contour') is not None:
                cv2.drawContours(result, [shape['contour']], -1, color, 3)
            else:
                cv2.circle(result, center, 30, color, 2)
            
            cv2.circle(result, center, 8, color, -1)
            cv2.circle(result, center, 11, (255, 255, 255), 2)
            cv2.line(result, origin, center, color, 1, cv2.LINE_AA)
            cv2.putText(result, str(i + 1), (center[0] - 8, center[1] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            text = f"{shape['type']}: ({shape['center_cm'][0]}, {shape['center_cm'][1]}) cm"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            
            text_x = center[0] + 40 if i % 2 == 0 else center[0] - tw - 40
            text_y = center[1] + 10
            
            cv2.rectangle(result,
                          (text_x - 8, text_y - th - 8),
                          (text_x + tw + 8, text_y + baseline + 8),
                          (0, 0, 0), -1)
            cv2.putText(result, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        return result

    def process(self):
        if not self.detect_main_square():
            return None, None
        
        if not self.detect_aruco_and_calibrate():
            return None, None
        
        self.shapes = self.detect_shapes_automatically()
        
        if len(self.shapes) != 4:
            print(f"\n⚠️  Detectadas {len(self.shapes)} formas (esperado: 4)")
            resposta = input("   Deseja marcar manualmente? (s/n): ").strip().lower()
            if resposta == 's':
                print("\n📍 Modo manual ativado")
                self.shapes = self.click_shape_centers()
                if self.shapes is None:
                    return None, None
        
        result_img = self.draw_final_result()
        
        print("\n" + "=" * 60)
        print("📊 COORDENADAS DAS FORMAS GEOMÉTRICAS")
        print("=" * 60)
        print(f"Quadrado principal: {self.main_size_cm:.0f}x{self.main_size_cm:.0f} cm")
        print(f"ArUco ID {ARUCO_ID}: {self.aruco_size_cm:.0f}x{self.aruco_size_cm:.0f} cm")
        print(f"Origem (0,0) = canto mais próximo do ArUco")
        print(f"Escala final: {self.scale_factor:.5f} cm/pixel")
        print(f"Eixo X+ (unit): ({self.eixo_x_vec[0]:.3f}, {self.eixo_x_vec[1]:.3f})")
        print(f"Eixo Y+ (unit): ({self.eixo_y_vec[0]:.3f}, {self.eixo_y_vec[1]:.3f})")
        print("(referencial do PAPEL, independente de rotação)\n")
        
        for i, shape in enumerate(self.shapes, 1):
            x, y = shape['center_cm']
            diam = shape.get('diam_cm', 0)
            diam_str = f" | Ø {diam:.1f} cm" if diam > 0 else ""
            print(f"{i}. {shape['type']:12} → X = {x:8.2f} cm | Y = {y:8.2f} cm{diam_str}")
        
        with open("coordenadas_formas.txt", "w", encoding="utf-8") as f:
            f.write("COORDENADAS DAS FORMAS GEOMÉTRICAS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Quadrado principal: {self.main_size_cm:.0f}x{self.main_size_cm:.0f} cm\n")
            f.write(f"ArUco ID {ARUCO_ID}: {self.aruco_size_cm:.0f}x{self.aruco_size_cm:.0f} cm\n")
            f.write(f"Origem (0,0) = canto mais próximo do ArUco\n")
            f.write(f"Escala final: {self.scale_factor:.5f} cm/pixel\n")
            f.write("Referencial: papel (independe de rotação da imagem)\n\n")
            for i, shape in enumerate(self.shapes, 1):
                diam = shape.get('diam_cm', 0)
                diam_str = f", Ø = {diam:.2f} cm" if diam > 0 else ""
                f.write(f"{i}. {shape['type']}: X = {shape['center_cm'][0]:.2f} cm, "
                        f"Y = {shape['center_cm'][1]:.2f} cm{diam_str}\n")
        
        print(f"\n✅ Coordenadas salvas em: coordenadas_formas.txt")
        
        return self.shapes, result_img


def main():
    print("=" * 60)
    print("🔷 DETECTOR AUTOMÁTICO DE FORMAS GEOMÉTRICAS")
    print("   Quadrado 80x80 + ArUco + Detecção HSV (azul)")
    print("   Origem adaptativa + Eixos baseados na rotação do ArUco")
    print("=" * 60)
    
    pasta_atual = os.getcwd()
    print(f"\n📁 Pasta atual: {pasta_atual}")
    
    arquivos = os.listdir(pasta_atual)
    imagens = [f for f in arquivos if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada!")
        return
    
    print("\n📸 Imagens disponíveis:")
    for i, img in enumerate(imagens, 1):
        print(f"   {i}. {img}")
    
    escolha = input("\n▶️  Escolha o número da imagem: ").strip()
    
    if not escolha.isdigit():
        print("❌ Entrada inválida!")
        return
    
    idx = int(escolha) - 1
    if idx < 0 or idx >= len(imagens):
        print("❌ Número inválido!")
        return
    
    image_path = imagens[idx]
    print(f"✓ Carregando: {image_path}")
    
    try:
        detector = HybridShapeDetector(image_path)
        shapes, result = detector.process()
        
        if shapes is None:
            print("\n❌ Operação cancelada!")
            return
        
        output = "resultado_final.jpg"
        cv2.imwrite(output, result)
        print(f"✅ Imagem salva como: {output}")
        
        height, width = result.shape[:2]
        scale = min(1.0, 1200 / width, 900 / height)
        if scale < 1.0:
            result_display = cv2.resize(result, (int(width * scale), int(height * scale)))
        else:
            result_display = result
        
        cv2.imshow("Resultado Final - Pressione qualquer tecla para fechar", result_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()