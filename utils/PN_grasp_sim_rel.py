import numpy as np
import os

grasp_sim_pred = np.load('/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/grasp_sim_preds.npy')
PN_op_pred = np.load('/home/vishaal/omniverse/new_1/pointnet.pytorch/utils/PN_op_preds.npy')

print(grasp_sim_pred.shape)
print(PN_op_pred.shape)

exps = np.linspace(-0.5, 2.5, num=11+20)

# # exp = 1.1
# for exp in exps:
#     cy = np.power(grasp_sim_pred, exp).copy()
#     num = np.sum(np.multiply(cy, PN_op_pred))
#     den = np.sum(cy) + np.sum(PN_op_pred)
#     print(exp, ':', num/den)


exp = 0.3
lists = sorted(os.listdir('/home/vishaal/Downloads/assets/dataset/grasping/test_objects'))

for i in range(grasp_sim_pred.shape[0]):
    # print(lists[i], ':', np.mean(abs(np.power(grasp_sim_pred[i], exp) - PN_op_pred[i])), ':', np.mean(abs(np.power(grasp_sim_pred[i], exp) - grasp_sim_pred[i])))
    cy = np.power(grasp_sim_pred, exp).copy()
    num = np.sum(np.multiply(cy[i],PN_op_pred[i]))
    den = np.sum(cy[i]) + np.sum(PN_op_pred[i])
    print(lists[i], ':', num/den)

          # print(np.mean(abs(np.power(grasp_sim_pred, exp) - grasp_sim_pred)))
# print(np.mean(abs(np.power(grasp_sim_pred, exp) - PN_op_pred)))