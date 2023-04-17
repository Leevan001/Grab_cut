def constructGCGraph(img, mask, bgdGMM, fgdGMM, lamda, leftW, upleftW, upW, uprightW):
    vtxCount = img.shape[0] * img.shape[1]
    # edgeCount = 2 * (4 * vtxCount - 3 * (img.shape[0] + img.shape[1]) + 2)
    graph = nx.Graph()

    graph.add_nodes_from(range(vtxCount))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = i * img.shape[1] + j
            color = img[i, j, :]

            if mask[i, j] == cv2.GC_PR_BGD or mask[i, j] == cv2.GC_PR_FGD:
                fromSource = -np.log(bgdGMM.predict_proba(color))
                toSink = -np.log(fgdGMM.predict_proba(color))

            elif mask[i, j] == cv2.GC_BGD:
                fromSource = lamda
                toSink = 0
            else:
                fromSource = 0
                toSink = lamda

            graph.add_edge('s', idx, capacity=fromSource)
            graph.add_edge(idx, 't', capacity=toSink)

            if j > 0:
                w = leftW[i, j]
                graph.add_edge(idx, idx - 1, capacity=w)

            if i > 0 and j > 0:
                w = upleftW[i, j]
                graph.add_edge(idx, idx - img.shape[1] - 1, capacity=w)
                graph.add_edge(idx - img.shape[1] - 1, idx, capacity=w)

            if i > 0:
                w = upW[i, j]
                graph.add_edge(idx, idx - img.shape[1], capacity=w)
                graph.add_edge(idx - img.shape[1], idx, capacity=w)
            if i > 0 and j < img.shape[1] - 1:
                w = uprightW[i, j]
                graph.add_edge(idx, idx - img.shape[1] + 1, capacity=w)
                graph.add_edge(idx - img.shape[1] + 1, idx, capacity=w)

    return graph