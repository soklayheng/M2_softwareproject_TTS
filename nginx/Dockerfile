FROM nginx

# WORKDIR /etc/nginx/conf.d
# 
# RUN rm ./default.conf
# 
# COPY nginx.conf ./
# 
# WORKDIR /

RUN rm /etc/nginx/conf.d/default.conf

COPY default.conf /etc/nginx/conf.d/
# COPY nginx.conf /etc/nginx/conf.d/