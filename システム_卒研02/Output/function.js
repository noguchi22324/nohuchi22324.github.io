async function js_to_py(a, b, c) {
  let result = await eel.add(a, b, c);
}

eel.expose(js_func)
function js_func(core_node_label, core_edge_prop,pf_uni_name,data3,core) {

  //ハッセ図の記述
  var width = 4000;
  var height = 3000;


  // ボタンに表示するラベルの配列
  var labels = core;
  // ボタンを表示する要素を取得
  var container = document.getElementById("buttons-container");
  // 配列の要素をラベルとするボタンを作成して要素に追加する
  labels.forEach(function(label) {
      var button = document.createElement("button"); // ボタン要素を作成
      button.innerHTML = label; // ボタンのラベルを設定
      button.addEventListener("click", function() {
          

          // inputArrayのデータを変更する
          //inputArray = core_node_label[1];
          //console.log("inputArrayの値が変更されました：" + inputArray);
          //alert("値" + inputArray + "が変更"); // ボタンがクリックされた時の処理
        });
      //container.appendChild(button); // ボタンを要素に追加
  });

  
  // nodeの定義。ここを増やすと楽しい。
  var inputArray = core_node_label[42]//+core_node_label[1];//pythonで計算した属性のラベルのリスト
  //var inputArray = core_node_label[28]
  //inputArray.push(...core_node_label[1]);
  //console.log(inputArray);

  
  var nodes = [];
  
  for (var i = 0; i < inputArray.length; i++) {
    var label = "";
    for (var j = 0; j < inputArray[i].length; j++) {
      // 各サブ配列の要素をlabelに追加
      label += inputArray[i][j];
      if (j < inputArray[i].length - 1) {
        label += "/"; // 要素間にスペースを追加（任意の区切り文字を使っても良い）
      }
    }
    nodes.push({ id: i, label: label });//{id ,labelを渡す}
  }
  console.log(nodes);
  

  // node同士の紐付け設定。実用の際は、ここをどう作るかが難しいのかも。[[親][子]]のセットで渡される
  var sourceData = core_edge_prop[42] //data3//core_edge_prop[0] //pythonで計算した属性のリスト
  //var sourceData = core_edge_prop[28]
  var links = [];
  
  for (var i = 0; i < sourceData[0].length; i++) {
    var source = sourceData[0][i];
    var target = sourceData[1][i];
    links.push({ source: source, target: target });
  }
  console.log(links);
  
  // forceLayout自体の設定はここ。ここをいじると楽しい。
  var force = d3.layout.force()
      .nodes(nodes)
      .links(links)
      .size([width, height])
      .distance(40) // node同士の距離 30
      .friction(0.9) // 摩擦力(加速度)的なものらしい。
      .charge(-4000) // 寄っていこうとする力。推進力(反発力)というらしい。ノードそれぞれが反発しやすい-4000
      .gravity(0.1) // 画面の中央に引っ張る力。引力。
      .start();

  // 描画場所を指定する要素を選択
  var visualizationDiv = d3.select("#main");
  // SVG要素を作成し、サイズを指定
  var svg = visualizationDiv.append("svg")
    .attr("width", width)
    .attr("height", height);


  // link線の描画(svgのline描画機能を利用)
  var link = svg.selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .style({stroke: "#b3b3b3",
      "stroke-width": 3
  });

  // ラベルと色の対応を定義
  var array1 = pf_uni_name;
  //var color_list = ['rgb(180, 230, 255)', 'rgb(180, 230, 255)', 'rgb(180, 230, 255)', 'rgb(180, 230, 255)', 'rgb(180, 230, 255)', 'rgb(180, 230, 255)'];
  //var color_list = ['rgb(230, 85, 115)', 'rgb(130, 220, 190)', 'rgb(120, 180, 255)', 'rgb(180, 230, 255)','rgb(255, 230, 130)', 'rgb(255, 230, 130)'];//図11用並び替えたver
  //var color_list = ['rgb(230, 85, 115)', 'rgb(130, 220, 190)', 'rgb(120, 180, 255)', 'rgb(255, 200, 200)','rgb(255, 200, 200)', 'rgb(180, 230, 255)'];//図13並び替えたver
  var color_list = ['rgb(230, 85, 115)', 'rgb(130, 220, 190)', 'rgb(120, 180, 255)', 'rgb(255, 200, 200)','rgb(255, 230, 130)', 'rgb(180, 230, 255)'];//並び替えたver
  //var color_list = ['rgb(240, 150, 170)', 'rgb(130, 220, 190)', 'rgb(120, 180, 255)', 'rgb(255, 230, 130)','rgb(180, 230, 255)', 'rgb(255, 200, 200)'];// 'rgb(200, 130, 200)', 'rgb(190, 160, 80)'];

  // インデックスに対応する辞書を作成
  var labelColorMap = {};
  for (var i = 0; i < pf_uni_name.length; i++) {
    labelColorMap[pf_uni_name[i]] = color_list[i];
  }

  // nodesの描画(今回はsvgの円描画機能を利用)
  var node = svg.selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr({
          // せっかくなので半径をランダムに
          r: function(d,i) {
            //if (i ==0) return 100;
            if (i ==0) return 100;
            else return 30;
          }
      })
      .style({
      // ノードのインデックスに基づいて色を設定
      fill: function(d, i) {
        if (i == 0) return "#cccccc";//コアはグレー
        else return labelColorMap[d.label] || "#cccccc";//決定クラスは色分けしてその他はグレー
      }
      })
      .call(force.drag);

  // nodeのラベル周りの設定
  //var label = svg.selectAll('text')
  var label = svg.selectAll('.label-text') // '.label-text'クラスを選択
      .data(nodes)
      .enter()
      .append('text')
      .attr({
          "text-anchor":"middle",
          "fill":"black",
          "font-size": "20px",
          "dy": ".35em" // テキストの中央揃えのための調整
      });
      //.text(function(data) { return data.label; });

// ラベルを ":" と "/" で分割して4行表示する
label.each(function(data) {
  var words;
  var lineHeight = 25; // 行間の高さ

  // ":" で分割する
  if (data.label.includes("/")) {
    words = data.label.split("/");
  }
  // "/" で分割する
  else if (data.label.includes(":")) {
    words = data.label.split(":");
  }
  // 分割する文字がない場合はそのまま表示
  else {
    words = [data.label];
  }

  // 最初の部分を表示
  var tspan1 = d3.select(this).append("tspan")
      .text(words[0])
      .attr("x", 0)
      .attr("dy", "0");

  // 残りの部分を次の行に表示
  if (words.length > 1) {
    var tspan2 = d3.select(this).append("tspan")
        .text(words.slice(1).join("/")) // 残りの部分を結合
        .attr("x", 0) // Xオフセットをリセット
        .attr("dy", lineHeight); // 次の行に移動
  }
});


  // tickイベント(力学計算が起こるたびに呼ばれるらしいので、座標追従などはここで)
  force.on("tick", function() {
      link.attr({
          x1: function(data) { return data.source.x;},
          y1: function(data) { return data.source.y;},
          x2: function(data) { return data.target.x;},
          y2: function(data) { return data.target.y;}
      });
      node.attr({
          cx: function(data) { return data.x;},
          cy: function(data) { return data.y;}
      });

      // labelも追随するように
      //label.attr("transform", function(data) {
      //  return "translate(" + data.x + "," + (data.y + 20) + ")"; // 20はノードの下に配置するためのオフセット
      //});

      // labelの位置を調整
      label.attr("transform", function(data) {
        var labelHeight = this.getBBox().height; // ラベルの高さを取得

        return "translate(" + data.x + "," + (data.y - labelHeight/8) + ")"; // ラベルの下に配置する
      });
      
      //label.attr({
      //    x: function(data) { return data.x;},
      //    y: function(data) { return data.y;}
      //});
  }); 
  


}
