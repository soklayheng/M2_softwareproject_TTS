FROM node:13.12.0-alpine

WORKDIR /app

COPY package.json .
COPY package-lock.json .
RUN npm install --silent

COPY . .

# RUN npm install -g serve

CMD ["npm", "start"]
# CMD serve -s build