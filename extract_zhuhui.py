import os
import sqlite3
conn = sqlite3.connect("C:/Users/zhaofeng/Apple/MobileSync/Backup/00008030-000C64C63446402E/6a/6ae5cb798aec8bf1e444ce8d19f0a961469f884f")
cursor = conn.cursor()
statement = 'SELECT CreateTime,Message,Des,Type,MesLocalID FROM Chat_a008711ad1fdc9f567f38778552cbbd3 limit 10;'
cursor.execute(statement)

displayname = '赵丰与朱慧'
html = "<!DOCTYPE html PUBLIC ""-//W3C//DTD XHTML 1.0 Transitional//EN"" ""http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"">"
html += "<html xmlns=""http://www.w3.org/1999/xhtml""><head><meta http-equiv=""Content-Type"" content=""text/html; charset=utf-8"" /><title>" + displayname + " - 微信聊天记录</title></head>"
html += "<body><table width=""600"" border=""0"" style=""font-size:12px;border-collapse:separate;border-spacing:0px 20px;word-break:break-all;table-layout:fixed;word-wrap:break-word;"" align=""center"">"

my_portaint = 'Feishu20220613-201140'
target_portaint = 'Feishu20220613-201131.jpg'
myself_DisplayName = '赵丰'
friend_DisplayName = '朱慧'
_id = 'zh2512382436'
root_dir = 'C:/Users/zhaofeng/Desktop/fc1faa899a951bf80f7755c8e40ca392/zhuhui/'

for entry in cursor.fetchall():
    unixtime = entry[0]
    message = entry[1]
    des = entry[2]
    type = entry[3]
    msgid = entry[4]
    print(message)

                           
    if type == 10000:
        html += "<tr><td width=""80"">&nbsp;</td><td width=""100"">&nbsp;</td><td>系统消息: " + message + "</td></tr>"
        continue
    ts = ""
            
    if des == 0:
        ts += "<tr><td width=""80"" align=""center""><img src=""Portrait/" + my_portaint +  ' width="50" height="50" /><br />' + myself_DisplayName + "</td>"
    else:
        ts += "<tr><td width=""80"" align=""center""><img src=""Portrait/" + target_portaint + ' width="50" height="50" /><br />' + friend_DisplayName + "</td>"
                
    if type == 34:
        audio_filename =  root_dir + _id + "_files/" + msgid + ".mp3"
        if not os.path.exists(audio_filename):
            message = "[语音]"
        else:
            message = "<audio controls><source src=\"" + _id + "_files/" + msgid + ".mp3\" type=\"audio/mpeg\"><a href=\"" + _id + "_files/" + msgid + ".mp3\">播放</a></audio>"
    elif type == 47:
    {
        var match = Regex.Match(message, @"cdnurl ?= ?""(.+?)""");
        if (match.Success)
        {
            var localfile = RemoveCdata( match.Groups[1].Value);
            var match2 = Regex.Match(localfile, @"\/(\w+?)\/\w*$");
            if (!match2.Success) localfile = RandomString(10);
            else localfile = match2.Groups[1].Value;
            emojidown.Add(new DownloadTask() { url = match.Groups[1].Value, filename = localfile + ".gif" });
            message = "<img src=\"Emoji/" + localfile + ".gif\" style=\"max-width:100px;max-height:60px\" />";
        }
        else message = "[表情]";
    }
    else if (type == 62 || type == 43)
    {
        var hasthum = RequireResource(MyPath.Combine(userBase, "Video", table, msgid + ".video_thum"), Path.Combine(assetsdir, msgid + "_thum.jpg"));
        var hasvid = RequireResource(MyPath.Combine(userBase, "Video", table, msgid + ".mp4"), Path.Combine(assetsdir, msgid + ".mp4"));
        if (hasthum && hasvid) message = "<video controls poster=\"" + id + "_files/" + msgid + "_thum.jpg\"><source src=\"" + id + "_files/" + msgid + ".mp4\" type=\"video/mp4\"><a href=\"" + id + "_files/" + msgid + ".mp4\">播放</a></video>";
        else if (hasthum) message = "<img src=\"" + id + "_files/" + msgid + "_thum.jpg\" /> （视频丢失）";
        else if (hasvid) message = "<video controls><source src=\"" + id + "_files/" + msgid + ".mp4\" type=\"video/mp4\"><a href=\"" + id + "_files/" + msgid + ".mp4\">播放</a></video>";
        else message = "[视频]";
    }
    else if (type == 50) message = "[视频/语音通话]";
    else if (type == 3)
    {
        var hasthum = RequireResource(MyPath.Combine(userBase, "Img", table, msgid + ".pic_thum"), Path.Combine(assetsdir, msgid + "_thum.jpg"));
        var haspic = RequireResource(MyPath.Combine(userBase, "Img", table, msgid + ".pic"), Path.Combine(assetsdir, msgid + ".jpg"));
        if (hasthum && haspic) message = "<a href=\"" + id + "_files/" + msgid + ".jpg\"><img src=\"" + id + "_files/" + msgid + "_thum.jpg\" style=\"max-width:100px;max-height:60px\" /></a>";
        else if (hasthum) message = "<img src=\"" + id + "_files/" + msgid + "_thum.jpg\" style=\"max-width:100px;max-height:60px\" />";
        else if (haspic) message = "<img src=\"" + id + "_files/" + msgid + ".jpg\" style=\"max-width:100px;max-height:60px\" />";
        else message = "[图片]";
    }
    else if (type == 48)
    {
        var match1 = Regex.Match(message, @"x ?= ?""(.+?)""");
        var match2 = Regex.Match(message, @"y ?= ?""(.+?)""");
        var match3 = Regex.Match(message, @"label ?= ?""(.+?)""");
        if (match1.Success && match2.Success && match3.Success) message = "[位置 (" + RemoveCdata( match2.Groups[1].Value) + "," + RemoveCdata(match1.Groups[1].Value) + ") " + RemoveCdata(match3.Groups[1].Value) + "]";
        else message = "[位置]";
    }
    else if (type == 49)
    {
        if (message.Contains("<type>2001<")) message = "[红包]";
        else if (message.Contains("<type>2000<")) message = "[转账]";
        else if (message.Contains("<type>17<")) message = "[实时位置共享]";
        else if (message.Contains("<type>6<"))
        {
            var match1 = Regex.Match(message, @"<fileext>(.+?)<\/fileext>");
            var match2 = Regex.Match(message, @"<title>(.+?)<\/title>");
            if (match1.Success && match2.Success)
            {
                var hasfile = RequireResource(MyPath.Combine(userBase, "OpenData", table, msgid + "." + match1.Groups[1].Value), Path.Combine(assetsdir, match2.Groups[1].Value));
                if (hasfile) message = "<a href=\"" + id + "_files/" + match2.Groups[1].Value + "\">" + match2.Groups[1].Value + "</a>";
                else message = match2.Groups[1].Value + "(文件丢失)";
            }
            else
            {
                message = "[文件]";
            }
        }
        else
        {
            var match1 = Regex.Match(message, @"<title>(.+?)<\/title>");
            var match2 = Regex.Match(message, @"<des>(.*?)<\/des>");
            var match3 = Regex.Match(message, @"<url>(.+?)<\/url>");
            var match4 = Regex.Match(message, @"<thumburl>(.+?)<\/thumburl>");
            if (match1.Success && match3.Success)
            {
                message = "";
                if (match4.Success) message += "<img src=\"" + RemoveCdata(match4.Groups[1].Value) + "\" style=\"float:left;max-width:100px;max-height:60px\" />";
                message += "<a href=\"" + RemoveCdata(match3.Groups[1].Value) + "\"><b>" + RemoveCdata(match1.Groups[1].Value) + "</b></a>";
                if (match2.Success) message += "<br />" + RemoveCdata(match2.Groups[1].Value);
            }
            else message = "[链接]";
        }
    }
    else if (type == 42)
    {
        var match1 = Regex.Match(message, "nickname ?= ?\"(.+?)\"");
        var match2=Regex.Match(message, "smallheadimgurl ?= ?\"(.+?)\"");
        if (match1.Success)
        {
            message = "";
            if(match2.Success)message+= "<img src=\"" + RemoveCdata(match2.Groups[1].Value) + "\" style=\"float:left;max-width:100px;max-height:60px\" />";
            message += "[名片] " + RemoveCdata(match1.Groups[1].Value);
        }
        else message = "[名片]";
    }
    else message = SafeHTML(message)

    ts += "<td width=""100"" align=""center"">" + FromUnixTime(unixtime).ToLocalTime().ToString().Replace(" ","<br />") + "</td>"
    ts += "<td>" + message + "</td></tr>"
    html += ts

html += "</body></html>"
