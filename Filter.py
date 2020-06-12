class Filter:
    def __init__(self):
        self.pre_trans_x = None
        self.pre_trans_y = None
        self.pre_trans_z = None
        
    def update(self, tvecs) -> bool:
        trans_x, trans_y, trans_z = tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2]
        is_mark_move = False
        if self.pre_trans_x is not None:
            if abs(self.pre_trans_x - trans_x) > 0.001 or abs(self.pre_trans_y - trans_y) > 0.002 or abs(self.pre_trans_z - trans_z) > 0.015:
                dis_x = abs(self.pre_trans_x - trans_x)
                dis_y = abs(self.pre_trans_y - trans_y)
                dis_z = abs(self.pre_trans_z - trans_z)
                # if dis_x > 0.001:
                #     print('dis_x', dis_x)
                # if dis_y > 0.001:
                #     print("dis_y", dis_y)
                # if dis_z > 0.001:
                #     print("dis_z", dis_z)
                
                is_mark_move = True
        self.pre_trans_x, self.pre_trans_y, self.pre_trans_z = trans_x, trans_y, trans_z
        return is_mark_move