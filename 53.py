#see question description here:https://leetcode-cn.com/problems/maximum-subarray/

class Solution:
    def maxSubArray(self, nums):
        cur_sum = 0
        res = nums[0]
        n = len(nums)
        for i in range(n):
            if cur_sum<0:
                cur_sum = nums[i]
            else:
                cur_sum+= nums[i]
            res = max(res,cur_sum)
        return res
a = Solution()
nums = [-1,3,4,-1]
print(a.maxSubArray(nums))



