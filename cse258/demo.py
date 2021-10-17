# 'E.164':
# may start with +, have up to 15 digits(without +)
# Digits will be between 0-9, with the first digit not be 0

# "channel"
# no whitespace
# provider:identifier, whatsapp, wechat and messenger can be valid provider
# identifier can be 1-248, and must be E.164 number for whatsapp and messenger
# Other providers can have identifiers that made up of any english alpha and +, - ,_ , @, .


import re


class Solution():
    def is_match(self, str):
        def is_E164(str):
            if str[0] == '+':
                str = str[1:]
            if str[0] == '0' or len(str) > 15:
                return False
            p = re.compile('^[0-9]*$')
            # for e in str:
            #     if p.match(e) is False:
            #         return False
            return p.match(str)

        if is_E164(str):
            return 'SMS'
        else:
            if str.find(':') == -1 or str.find(' ') != -1:
                return False
            str = str.split(':')
            if len(str) >= 3:
                return False
            if str[0] == 'whatsapp' or str[0] == 'messenger':
                return str[0] if is_E164(str[1]) else False
            elif str[0] == 'wechat':
                if len(str[1]) > 248:
                    return False



if __name__ == '__main__':
    s1 = 'asdfa:2342 4'
    s2 = 'whatsapp:123'
    s3 = 'wechat:gh_a1f37dacefe3'

    s = Solution()
    print(s.is_match(s1))
    print(s.is_match(s2))
    print(s.is_match(s3))
