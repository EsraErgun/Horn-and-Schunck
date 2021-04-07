predict_from_previous_predicted = True

def read(image):

    if isinstance(image, str):
    
        Image = cv2.imread(image)
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
       
    elif isinstance(image, np.ndarray):
        
        Image = anchor 
    else:
        raise ValueError

    return (Image)

def horn_and_schunck(anchorPath, targetPath, Lambda, n_iter, error = False):
    
    '''
    Input: anchor image and target image.
    Return: motion_vectors
    
    '''
    if error == True:
        
        error_arr = list()
        
    anchorFrame, targetFrame = read(anchorPath), read(targetPath)
    anchorFrame, targetFrame = anchorFrame.astype(np.float32), targetFrame.astype(np.float32)
    h, w = int(anchorFrame.shape[0]), int(anchorFrame.shape[1])    
    average_filter = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

    filter_x = np.array([[-1, 1],
                        [-1, 1]]) * .25  

    filter_y = np.array([[-1, -1],
                        [1, 1]]) * .25  

    
    filter_t = np.ones((2, 2))*.25
    from scipy import ndimage
    fx = ndimage.convolve(anchorFrame, filter_x) + ndimage.convolve(targetFrame, filter_x)
    #print(fx)
    fy = ndimage.convolve(anchorFrame, filter_y) + ndimage.convolve(targetFrame, filter_y)
    ft = ndimage.convolve(anchorFrame, filter_t) + ndimage.convolve(targetFrame, -filter_t) 
    v_x, v_y = np.zeros((h,w)), np.zeros((h, w))
    
    
    for i in range(n_iter):
        v_x_diff = 0.25*(ndimage.convolve(v_x, filter_x)**2) + 0.25*(ndimage.convolve(v_x, filter_y)**2)
        v_y_diff = 0.25*(ndimage.convolve(v_y, filter_x)**2) + 0.25*(ndimage.convolve(v_y, filter_y)**2)
        if error == True:
            error_term = Lambda * (v_x_diff+v_y_diff) + (fx*v_x + fy*v_y + ft)**2
            error_arr.append(error_term)
        v_x_av = ndimage.convolve(v_x, average_filter)
        v_y_av = ndimage.convolve(v_y, average_filter)
        v_x = v_x_av - fx*((fx*v_x_av+fy*v_y_av+ft)/(Lambda**2 + fx**2 + fy**2))
        v_y = v_y_av - fy*((fx*v_x_av+fy*v_y_av+ft)/(Lambda**2 + fx**2 + fy**2))

    if error == True:
        return((v_x, v_y, error_arr))
    else:
        return((v_x, v_y))
    

def generate_target_frame(anchorPath, U, V,path, interpolation = cv2.INTER_LINEAR):
    anchorFrame = read(anchorPath).astype('uint8')
    #anchorFrame = cv2.imread(anchorPath, cv2.IMREAD_COLOR).astype('uint8')
    print(anchorFrame.shape)
    h, w = U.shape[0], U.shape[1]
    #targetFrame = np.zeros((h, w))
       
    #src = cv2.imread(cv2.samples.findFile(args.input), cv2.IMREAD_COLOR)
    U +=np.arange(w)
    V +=np.arange(h)[:,np.newaxis]
    targetFrame = cv2.remap(anchorFrame, U, V, interpolation)  
    print(targetFrame.shape)
    cv2.imwrite(path, targetFrame)
    
def generate_color_coded_vec(u, v, path):
    
    import flow_vis

    U_e= np.expand_dims(u, axis = 2)
    V_e = np.expand_dims(v, axis = 2)
    U_e_norm = np.linalg.norm(U_e)
    V_e_norm = np.linalg.norm(V_e)
    U_e = U_e*2/U_e_norm
    V_e = V_e*2/V_e_norm
    UV = np.concatenate((U_e, V_e), axis = 2)

    flow_color = flow_vis.flow_to_color(UV, convert_to_bgr=False)

    plt.imshow(flow_color)
    plt.savefig(path)
